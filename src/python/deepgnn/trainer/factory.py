"""New trainer base."""

import os
import argparse
import time
from typing import Optional, Dict, Callable, List, Tuple, IO

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data.distributed
import horovod.torch as hvd

from ray import train
from ray.air import session
from ray.air.config import ScalingConfig
from ray.train.torch import TorchTrainer
from ray.train.tensorflow import TensorflowTrainer
from ray.train.horovod import HorovodTrainer

from deepgnn import get_logger
from deepgnn.pytorch.modeling.base_model import BaseModel
from deepgnn.pytorch.common import init_common_args
from deepgnn.trainer.args import init_trainer_args, init_fp16_args
from deepgnn.graph_engine import create_backend, BackendOptions
from deepgnn.graph_engine.samplers import GENodeSampler, GEEdgeSampler
from deepgnn.pytorch.common.dataset import TorchDeepGNNDataset
from deepgnn.pytorch.common.consts import PREFIX_CHECKPOINT, PREFIX_EMBEDDING
from deepgnn.pytorch.common.utils import (
    dump_gpu_memory,
    print_model,
    tally_parameters,
    rotate_checkpoints,
    get_sorted_checkpoints,
    to_cuda,
)
from deepgnn.pytorch.common.consts import FP16_APEX, FP16_AMP, FP16_NO
from deepgnn import (
    log_telemetry,
    TrainerType,
    TrainMode,
    LOG_PROPS_EVENT_START_WORKER,
    LOG_PROPS_PLATFORM_PYTORCH,
    LOG_PROPS_EVENT_END_WORKER,
)
from deepgnn.pytorch.common.optimization import get_linear_schedule_with_warmup
from deepgnn.graph_engine.adl_uploader import AdlDataWriter


class DeepGNNTrainingLoop:
    """
    Pytorch trainer controls the workflow of training/evaluation/inference.

    - Implementation in this class only works for sinle worker FP32 training requirement.
    - For FP16 mixed precision training, please use FP16Trainer.
    - For distributed training, please use DDPTrainer or HVDTrainer.
    """

    def __init__(self):
        """Initialize trainer."""

    def run(self, config):
        """
        Perform training/evaluation/inference according to training mode set in constructor.

        Args:
            model: target model.
            dataset: dataset for training, evaluation or inference.
            optimizer: optimizer for training.
            eval_during_train_dataset: optional dataset for evaluation
                during training.
        """
        self.args = config["args"]
        self._initialize(
            config["init_model_fn"], config["init_optimizer_fn"], config["init_dataset_fn"], config["init_eval_dataset_for_training_fn"]
        )

        log_telemetry(
            self.logger,
            f"Training worker started. Model: {self.model_name}.",
            LOG_PROPS_EVENT_START_WORKER,
            f"{self.args.mode}",
            self.model_name,
            self.args.user_name,
            self.args.job_id,
            self.rank,
            self.world_size,
            LOG_PROPS_PLATFORM_PYTORCH,
        )

        result = None
        if self.args.mode == TrainMode.TRAIN:
            assert self.optimizer is not None
            self._train(self.model)
        elif self.args.mode == TrainMode.EVALUATE:
            result, loss = self._evaluate(self.model)
        elif self.args.mode == TrainMode.INFERENCE:
            self._inference(self.model)
        else:
            raise RuntimeError(f"Unsupported TrainMode:{self.args.mode}")

        log_telemetry(
            self.logger,
            f"Training worker finished. Model: {self.model_name}.",
            LOG_PROPS_EVENT_END_WORKER,
            f"{self.args.mode}",
            self.model_name,
            self.args.user_name,
            self.args.job_id,
            self.rank,
            self.world_size,
            LOG_PROPS_PLATFORM_PYTORCH,
        )
        if self.args.gpu:
            self.logger.info(self._wrap_log(dump_gpu_memory()))

        return result

    def _initialize(
        self,
        init_model_fn: BaseModel,
        init_optimizer_fn: TorchDeepGNNDataset,  # TODO optimizer fn optional
        init_dataset_fn: Optional[Optimizer],
        init_eval_dataset_for_training_fn: TorchDeepGNNDataset = None,
    ) -> BaseModel:
        """Initialize all training loop variables."""
        hvd.init()
        torch.manual_seed(self.args.seed)
        torch.set_num_threads(1)  # Horovod: limit # of CPU threads to be used per worker.

        self.logger = get_logger()

        # Initialize trainer state.
        self.step = 0
        self.global_step = 0
        self.epochs_trained = 0
        self.steps_in_epoch_trained = 0
        self.start_time = time.time()

        # Initialize rank, local_rank, world_size.
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        if self.args.trainer == TrainerType.HVD:
            self.local_rank = hvd.local_rank()

        self.max_steps = (
            self.args.max_samples // (self.world_size * self.args.batch_size)
            if self.args.max_samples > 0
            else -1
        )
        self.logger.info(self._wrap_log(f"Max steps per epoch:{self.max_steps}"))

        self.model = self._init_model(init_model_fn)
        self.optimizer = self._init_optimizer(init_optimizer_fn)
        self.dataset = self._init_dataset(init_dataset_fn)
        self.eval_dataset_for_training = self._init_dataset(init_eval_dataset_for_training_fn)

    def _init_model(self, init_model_fn: callable) -> BaseModel:
        model = init_model_fn(self.args)
        self.model_name = type(model).__name__
        self.model_unwrapped = model
        if self.args.trainer == TrainerType.BASE:
            model = train.torch.prepare_model(model)
        elif self.args.trainer == TrainerType.HVD:
            # TODO necessary?
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)

        if self.args.gpu:
            torch.cuda.set_device(self.local_rank)
            torch.cuda.manual_seed(self.args.seed)
            model.cuda()

        if self.rank == 0:
            if (
                not self.args.enable_adl_uploader
                or self.args.mode != TrainMode.INFERENCE
            ):
                os.makedirs(self.args.save_path, exist_ok=True)
            print_model(model)
            tally_parameters(model)

        self._load_checkpoint()
        return model

    def _init_optimizer(self, init_optimizer_fn: Optimizer) -> Optimizer:
        if not init_optimizer_fn:
            return None

        optimizer = init_optimizer_fn(
            args=self.args, model=self.model, world_size=self.world_size
        )
        if self.args.trainer == TrainerType.HVD:
            # TODO necessary?
            #hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
            compression = (
                hvd.Compression.fp16 if self.fp16_enabled() else hvd.Compression.none
            )
            self.optimizer = hvd.DistributedOptimizer(
                self.optimizer,
                named_parameters=self.model.named_parameters(),
                compression=compression,
                op=hvd.Average,
            )

        self.lr_scheduler = None
        if self.max_steps > 0:
            #lr_scaler = hvd.size() if not use_adasum else 1
            num_training_steps = self.max_steps * self.args.num_epochs
            self.lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup * num_training_steps,
                num_training_steps=num_training_steps,
            )
        return optimizer

    def _init_dataset(self, init_dataset_fn: callable) -> Optimizer:
        if not init_dataset_fn:
            return None
        if not hasattr(self, "backend") or self.backend is None:
            self.backend = create_backend(
                BackendOptions(self.args), is_leader=(self.rank == 0)
            )
        dataset = init_dataset_fn(
            args=self.args,
            model=self.model_unwrapped,
            rank=self.rank,
            world_size=self.world_size,
            backend=self.backend,
        )
        num_workers = (
            0
            if issubclass(dataset.sampler_class, (GENodeSampler, GEEdgeSampler))
            else self.args.data_parallel_num
        )
        if True:
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                num_workers=num_workers,
                prefetch_factor=self.args.prefetch_factor,
            )
        else:
            # TODO no compatibile with iterable dataset, requires indexing
            dataset._torch_init_sampler()
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=hvd.size(), rank=hvd.rank()
            )
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                num_workers=num_workers,
                batch_size=self.args.batch_size,
                #prefetch_factor=self.args.prefetch_factor,
                sampler=sampler,
            )
        return dataloader

    def _train(self, model: BaseModel):
        self._init_summary_writer(prefix="train/worker")
        model.train()

        # Continous training from epoch and step saved in checkpoint.
        self.step = self.steps_in_epoch_trained
        for epoch in range(self.epochs_trained, self.args.num_epochs):
            if hasattr(self, "sampler"):
                self.sampler.set_epoch(epoch)
            for i, data in enumerate(self.dataset):
                # Skip trained steps.
                if i < self.step:
                    continue

                self.train_losses = []  # type: List[float]
                self.train_metrics = []  # type: List[float]
                self._increment_step()
                self._prepare_data(data)

                self.optimizer.zero_grad()
                loss, pred, label = model(data)
                # TODO loss caluclation here
                loss.backward()
                if self.args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.args.grad_max_norm
                    )
                self.optimizer.step()
                self.train_losses.append(loss.data.item())

                if self.lr_scheduler:
                    self.lr_scheduler.step()

                if (
                    self.rank == 0
                    and self.args.save_ckpt_by_steps > 0
                    and self.step % self.args.save_ckpt_by_steps == 0
                ):
                    self._save_checkpoint(epoch)

                # Calculate training metrics for one batch data.
                metric = (  # move to model.get_metric(*args)
                    self.model_unwrapped.compute_metric([pred], [label]).data.item()
                    if self.args.use_per_step_metrics
                    else torch.tensor(0.0)
                )
                self.train_metrics.append(metric)

                if (
                    self.args.log_by_steps > 0
                    and self.step % self.args.log_by_steps == 0
                ):
                    train_loss = np.mean(self.train_losses)
                    train_metric = np.mean(self.train_metrics)
                    self.train_losses = []
                    self.train_metrics = []
                    duration = self._check_duration()
                    self.summary_writer.add_scalar(
                        "Training/Loss", train_loss, self.global_step
                    )
                    self.summary_writer.add_scalar(
                        "Training/Time", duration, self.global_step
                    )
                    if self.args.use_per_step_metrics:
                        self.summary_writer.add_scalar(
                            f"Training/{self.model_unwrapped.metric_name()}",
                            train_metric,
                            self.global_step,
                        )
                    if self.lr_scheduler:
                        self.summary_writer.add_scalar(
                            "Training/LR",
                            self.lr_scheduler.get_last_lr()[0],
                            self.global_step,
                        )

                    self.logger.info(
                        self._wrap_log(
                            f"epoch: {epoch}; step: {self.step:05d}; loss: {train_loss:.4f};"
                            + (
                                f" {self.model_unwrapped.metric_name()}: {train_metric:.4f}; time: {duration:.4f}s"
                                if self.args.use_per_step_metrics
                                else f" time: {duration:.4f}s"
                            )
                        )
                    )

                if (
                    self.eval_dataset_for_training is not None
                    and self.args.eval_during_train_by_steps > 0
                    and self.step % self.args.eval_during_train_by_steps == 0
                ):
                    model.eval()
                    eval_metric, eval_loss = self._evaluate(model)
                    if self.args.use_per_step_metrics:
                        self.summary_writer.add_scalar(
                            f"Validation/{self.model_unwrapped.metric_name()}",
                            eval_metric,
                            self.global_step,
                        )
                        self.summary_writer.add_scalar(
                            "Validation/Loss", eval_loss, self.global_step
                        )
                    self.logger.info(
                        self._wrap_log(
                            f"epoch: {epoch}; step: {self.step:05d};"
                            + (
                                f" Validation/{self.model_unwrapped.metric_name()}: {eval_metric:.4f}, Validation/Loss: {eval_loss:.4f}"
                                if self.args.use_per_step_metrics
                                else ""
                            )
                        )
                    )
                    model.train()

                if self._should_stop():
                    break

            # Epoch finishes.
            self.step = 0
            if self.rank == 0 and epoch % self.args.save_ckpt_by_epochs == 0:
                self._save_checkpoint(epoch + 1)
        session.report(dict(loss=loss))
        save_path = os.path.join(
            f"{self.args.save_path}",
            f"{PREFIX_CHECKPOINT}.pt",
        )
        torch.save(
            {"state_dict": self.model.state_dict(), "epoch": epoch, "step": self.step},
            save_path,
        )

        """
        # Horovod: use train_sampler to determine the number of
        # examples in this worker's partition.
        print(
            "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,
                batch_idx * len(data),
                len(train_sampler),
                100.0 * batch_idx / len(train_loader),
                loss.item(),
            )
        )
        """

    def _evaluate(self, model: BaseModel) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.args.mode != TrainMode.TRAIN:
            self._init_summary_writer(prefix="evaluate/worker")
        model.eval()
        preds = []
        labels = []
        eval_metrics = []
        eval_losses = []
        data_size = 0
        is_eval_during_training = self.args.mode == TrainMode.TRAIN
        dataset = (
            self.eval_dataset_for_training
            if is_eval_during_training
            else self.dataset
        )
        assert dataset is not None
        with torch.no_grad():
            for data in dataset:
                if not is_eval_during_training:
                    self._increment_step()
                self._prepare_data(data)

                loss, pred, label = model(data)

                if (
                    not is_eval_during_training
                    and self.args.log_by_steps > 0
                    and self.step % self.args.log_by_steps == 0
                ):
                    duration = self._check_duration()
                    loss_val = loss.data.item()
                    self.summary_writer.add_scalar(
                        "Evaluation/Loss", loss_val, self.global_step
                    )
                    self.summary_writer.add_scalar(
                        "Evaluation/Time", duration, self.global_step
                    )
                    self.logger.info(
                        self._wrap_log(
                            f"step: {self.step:05d}; loss: {loss_val:.4f}; time: {duration:.4f}s"
                        )
                    )

                data_size += pred.size(0)
                eval_losses.append(loss.data.item())

                if self.args.use_per_step_metrics:
                    metric = self.model_unwrapped.compute_metric([pred], [label])
                    eval_metrics.append(metric.data.item())
                else:
                    preds.append(pred.detach().cpu())
                    labels.append(label.detach().cpu())

                if self._should_stop():
                    break

            if self.args.use_per_step_metrics:
                eval_metric = np.mean(eval_metrics) if len(eval_metrics) > 0 else 0
                eval_metric = torch.tensor(eval_metric)
            else:
                eval_metric = self.model_unwrapped.compute_metric(preds, labels)
            self.logger.info(
                self._wrap_log(
                    f"Evaluation {self.model_unwrapped.metric_name()}: {eval_metric:.4f}; data size: {data_size};"
                )
            )
            eval_loss = np.mean(eval_losses) if len(eval_losses) > 0 else 0
            eval_loss = torch.tensor(eval_loss)
        """
        # HOROVOD eval metric
        metric, loss = super()._evaluate(model)
        metric = hvd.allreduce(metric)
        loss = hvd.allreduce(loss)
        self.logger.info(
            self._wrap_log(
                f"AllReduced {self.model_unwrapped.metric_name()}: {metric:.4f}; loss: {loss:.4f}"
            )
        )
        """
        return eval_metric, eval_loss

    def _inference(self, model: BaseModel):
        self._init_summary_writer(prefix="inference/worker")
        model.eval()

        with self._get_embedding_writer() as fp:
            with torch.no_grad():
                for data in self.dataset:
                    self._increment_step()
                    self._prepare_data(data)

                    embedding = self.model.get_embedding(data)
                    self.model.output_embedding(fp, data, embedding)

                    if (
                        self.args.log_by_steps > 0
                        and self.step % self.args.log_by_steps == 0
                    ):
                        duration = self._check_duration()
                        self.summary_writer.add_scalar(
                            "Inference/Time", duration, self.global_step
                        )
                        self.logger.info(
                            self._wrap_log(
                                f"step: {self.step:05d}; time: {duration:.4f}s"
                            )
                        )
                    if self._should_stop():
                        break

    def _increment_step(self):
        self.step += 1
        self.global_step += 1

    def _should_stop(self) -> bool:
        return self.max_steps > 0 and self.step >= self.max_steps

    def fp16_enabled(self) -> bool:
        """Check if trainer should use fp16 mode."""
        return self.args.gpu and self.args.fp16 != FP16_NO

    def _check_duration(self) -> float:
        duration = time.time() - self.start_time
        self.start_time = time.time()
        return duration

    def _prepare_data(self, data: Dict):
        if self.args.gpu:
            to_cuda(data)

    def _wrap_log(self, content: str) -> str:
        return f"[{self.world_size},{self.rank}] {content}"

    def _save_checkpoint(self, epoch: int):
        # Don't save for last step to avoid duplication with ckpt after epoch finished.
        if self.max_steps > 0 and self.step == self.max_steps:
            return

        save_path = os.path.join(
            f"{self.args.save_path}",
            f"{PREFIX_CHECKPOINT}-{epoch:03}-{self.step:06}.pt",
        )
        torch.save(
            {"state_dict": self.model.state_dict(), "epoch": epoch, "step": self.step},
            save_path,
        )
        self.logger.info(self._wrap_log(f"Saved checkpoint to {save_path}."))
        rotate_checkpoints(self.args.save_path, self.args.max_saved_ckpts)

    def _load_checkpoint(self, ckpt_path: str = None):
        if not ckpt_path:
            # Search and sort checkpoints from model path.
            ckpts = get_sorted_checkpoints(self.args.model_dir)
            ckpt_path = ckpts[-1] if len(ckpts) > 0 else None

        if ckpt_path is not None:

            init_ckpt = torch.load(ckpt_path, map_location="cpu")
            self.epochs_trained = init_ckpt["epoch"]
            self.steps_in_epoch_trained = init_ckpt["step"]

            # Only worker with rank 0 loads state dict.
            if self.rank == 0:
                self.model.load_state_dict(init_ckpt["state_dict"])
                self.logger.info(
                    self._wrap_log(
                        f"Loaded initial checkpoint: {ckpt_path},"
                        f" trained epochs: {self.epochs_trained}, steps: {self.steps_in_epoch_trained}"
                    )
                )

            del init_ckpt

    def _init_summary_writer(self, prefix: str):
        self.summary_writer = SummaryWriter(
            os.path.join(self.args.metric_dir, f"{prefix}-{self.rank}")
        )

    def _get_embedding_writer(self) -> IO[str]:
        embed_path = os.path.join(
            self.args.save_path, f"{PREFIX_EMBEDDING}-{self.rank}"
        )
        if self.args.enable_adl_uploader:
            uploader = AdlDataWriter(
                process_num=self.args.uploader_process_num,
                threads_per_process=self.args.uploader_threads_num,
                queue_size=200,
                store_name=self.args.uploader_store_name,
                file_path_prefix=embed_path,
            )

            return uploader  # type: ignore
        return open(embed_path + ".tsv", "w", encoding="utf-8")


def get_args(init_arg_fn: Optional[Callable] = None, run_args: Optional[List] = None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Initialize common parameters, including model, dataset, optimizer etc.
    init_common_args(parser)

    # Initialize trainer paramaters.
    init_trainer_args(parser)

    # Initialize fp16 related paramaters.
    init_fp16_args(parser)

    if init_arg_fn is not None:
        init_arg_fn(parser)

    args = (
        parser.parse_known_args()[0]
        if run_args is None
        else parser.parse_args(run_args)
    )
    for arg in dir(args):
        if not arg.startswith("_"):
            get_logger().info(f"{arg}={getattr(args, arg)}")

    return args


def run_dist(
    init_model_fn: Callable,
    init_dataset_fn: Callable,
    init_optimizer_fn: Optional[Callable] = None,
    init_args_fn: Optional[Callable] = None,
    run_args: Optional[List] = None,
    init_eval_dataset_for_training_fn: Optional[Callable] = None,
):
    """Public api."""
    import ray

    ray.init(
        num_cpus=4, num_gpus=0, include_dashboard=False  # Need 2x num_workers
    )  # TODO how to set how many cpus each trainer is allocated
    args = get_args(init_args_fn, run_args)

    config = {
        "init_model_fn": init_model_fn,
        "init_dataset_fn": init_dataset_fn,
        "init_optimizer_fn": init_optimizer_fn,
        "init_args_fn": init_args_fn,
        "args": args,
        "init_eval_dataset_for_training_fn": init_eval_dataset_for_training_fn,
    }

    # TODO multi worker, each training worker should be looking at different partition of whole dataset
    # TODO add trainer worker rank value
    # TODO DeepGNNTrainingLooop init - start server?
    try:
        training_loop = DeepGNNTrainingLoop()  # DeepGNNTrainingLoop
        if args.trainer == TrainerType.BASE:
            if False:# TODO tf:
                trainer_class = TensorflowTrainer
            else:
                trainer_class = TorchTrainer
        elif args.trainer == TrainerType.HVD:
            trainer_class = HorovodTrainer
        elif args.trainer == TrainerType.DDP:
            trainer_class = None
        elif args.trainer == TrainerType.PS:
            if not tf:
                raise 
            trainer_class = None
        elif args.trainer == TrainerType.MULTINODE:
            if not tf:
                raise 
            trainer_class = None

        trainer = trainer_class(
            training_loop.run,
            train_loop_config=config,
            scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
        )
        results = trainer.fit()
    except Exception as e:
        ray.shutdown()
        raise e
    ray.shutdown()
    return results
