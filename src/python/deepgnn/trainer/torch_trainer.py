# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Trainer implementation for torch models."""
import argparse
import time
import torch
import os
import numpy as np

from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Optional, Dict, List

from deepgnn import (
    get_logger,
    log_telemetry,
    TrainMode,
    LOG_PROPS_EVENT_START_WORKER,
    LOG_PROPS_PLATFORM_PYTORCH,
    LOG_PROPS_EVENT_END_WORKER,
)
from deepgnn.pytorch.modeling.base_model import BaseModel
from deepgnn.pytorch.common.consts import PREFIX_CHECKPOINT, PREFIX_EMBEDDING
from deepgnn.pytorch.common.optimization import get_linear_schedule_with_warmup
from deepgnn.pytorch.common.utils import (
    dump_gpu_memory,
    print_model,
    tally_parameters,
    rotate_checkpoints,
    get_sorted_checkpoints,
    to_cuda,
)
from deepgnn.graph_engine.adl_uploader import AdlDataWriter


class Trainer:
    """
    Pytorch trainer controls the workflow of training/evaluation/inference.

    - Implementation in this class only works for sinle worker FP32 training requirement.
    - For FP16 mixed precision training, please use FP16Trainer.
    - For distributed training, please use DDPTrainer or HVDTrainer.
    """

    def __init__(self, args: argparse.Namespace):
        """Initialize trainer."""
        self.logger = get_logger()
        self.args = args
        # Initialize rank, local_rank, world_size.
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1

        # Initialize trainer state.
        self.step = 0
        self.global_step = 0
        self.epochs_trained = 0
        self.steps_in_epoch_trained = 0
        self.start_time = time.time()
        self.max_steps = 0

    def run(
        self,
        model: BaseModel,
        dataset: Any,
        optimizer: Optional[Optimizer] = None,
        eval_dataset_for_training: Any = None,
    ):
        """
        Perform training/evaluation/inference according to training mode set in constructor.

        Args:
            model: target model.
            dataset: dataset for training, evaluation or inference.
            optimizer: optimizer for training.
            eval_during_train_dataset: optional dataset for evaluation
                during training.
        """
        self._init_max_steps()
        model = self._initialize(model, dataset, optimizer, eval_dataset_for_training)

        # On-demand enable telemetry.
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
            self._train(model)
        elif self.args.mode == TrainMode.EVALUATE:
            result, loss = self._evaluate(model)
        elif self.args.mode == TrainMode.INFERENCE:
            self._inference(model)
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

    def _init_model(self, model: BaseModel):
        self.model = model
        self.model_name = type(self.model).__name__

        if self.args.gpu:
            torch.cuda.set_device(self.local_rank)
            self.model.cuda()

        if self.rank == 0:
            if (
                not self.args.enable_adl_uploader
                or self.args.mode != TrainMode.INFERENCE
            ):
                os.makedirs(self.args.save_path, exist_ok=True)
            print_model(self.model)
            tally_parameters(self.model)

        self._load_checkpoint()
        return model

    def _init_optimizer(self, optimizer: Optimizer):
        self.lr_scheduler = self._create_lr_scheduler(optimizer)
        return optimizer

    def _initialize(
        self,
        model: BaseModel,
        dataset: Any,
        optimizer: Optional[Optimizer] = None,
        eval_dataset_for_training: Any = None,
    ):
        model = self._init_model(model)
        self.dataset = dataset
        self.eval_dataset_for_training = eval_dataset_for_training
        self.optimizer = self._init_optimizer(optimizer) if optimizer else optimizer

        return model

    def _train(self, model: Module):
        self._init_summary_writer(prefix="train/worker")
        model.train()

        # Continous training from epoch and step saved in checkpoint.
        self.step = self.steps_in_epoch_trained
        for epoch in range(self.epochs_trained, self.args.num_epochs):
            self._train_one_epoch(model, epoch)

    def _train_one_epoch(self, model: Module, epoch: int):
        for i, data in enumerate(self.dataset):
            # Skip trained steps.
            if i < self.step:
                continue

            self.train_losses = []  # type: List[float]
            self.train_metrics = []  # type: List[float]
            self._train_one_step(model, data, epoch)
            if self._should_stop():
                break

        # Epoch finishes.
        self.step = 0
        if self.rank == 0 and epoch % self.args.save_ckpt_by_epochs == 0:
            self._save_checkpoint(epoch + 1)

    def _train_one_step(self, model: Module, data: Dict, epoch: int):
        self._increment_step()
        self._prepare_data(data)

        self.optimizer.zero_grad()
        loss, pred, label = self._train_one_step_internal(model, data)
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
        metric = (
            self.model.compute_metric([pred], [label]).data.item()
            if self.args.use_per_step_metrics
            else torch.tensor(0.0)
        )
        self.train_metrics.append(metric)  # TODO next PR, try to have ray track monitors/metrics

        if self.args.log_by_steps > 0 and self.step % self.args.log_by_steps == 0:
            train_loss = np.mean(self.train_losses)
            train_metric = np.mean(self.train_metrics)
            self.train_losses = []
            self.train_metrics = []
            duration = self._check_duration()
            self.summary_writer.add_scalar(
                "Training/Loss", train_loss, self.global_step
            )
            self.summary_writer.add_scalar("Training/Time", duration, self.global_step)
            if self.args.use_per_step_metrics:
                self.summary_writer.add_scalar(
                    f"Training/{self.model.metric_name()}",
                    train_metric,
                    self.global_step,
                )
            if self.lr_scheduler:
                self.summary_writer.add_scalar(
                    "Training/LR", self.lr_scheduler.get_last_lr()[0], self.global_step
                )

            self.logger.info(
                self._wrap_log(
                    f"epoch: {epoch}; step: {self.step:05d}; loss: {train_loss:.4f};"
                    + (
                        f" {self.model.metric_name()}: {train_metric:.4f}; time: {duration:.4f}s"
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
                    f"Validation/{self.model.metric_name()}",
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
                        f" Validation/{self.model.metric_name()}: {eval_metric:.4f}, Validation/Loss: {eval_loss:.4f}"
                        if self.args.use_per_step_metrics
                        else ""
                    )
                )
            )
            model.train()

    def _train_one_step_internal(self, model: Module, data: Dict):
        loss, pred, label = model(data)
        loss.backward()
        if self.args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_max_norm)
        self.optimizer.step()
        return loss, pred, label

    def _evaluate(self, model: Module):
        if self.args.mode != TrainMode.TRAIN:
            self._init_summary_writer(prefix="evaluate/worker")
        model.eval()

        preds = []
        labels = []
        eval_metrics = []
        eval_losses = []
        data_size = 0
        dataset = (
            self.eval_dataset_for_training
            if self.args.mode == TrainMode.TRAIN
            else self.dataset
        )
        assert dataset is not None
        with torch.no_grad():
            for data in dataset:
                pred, label, loss = self._evaluate_one_step(model, data)
                data_size += pred.size(0)
                eval_losses.append(loss.data.item())

                if self.args.use_per_step_metrics:
                    metric = self.model.compute_metric([pred], [label])
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
            eval_metric = self.model.compute_metric(preds, labels)
        self.logger.info(
            self._wrap_log(
                f"Evaluation {self.model.metric_name()}: {eval_metric:.4f}; data size: {data_size};"
            )
        )
        eval_loss = np.mean(eval_losses) if len(eval_losses) > 0 else 0
        eval_loss = torch.tensor(eval_loss)
        return eval_metric, eval_loss

    def _evaluate_one_step(self, model: Module, data: Dict):
        is_eval_during_training = self.args.mode == TrainMode.TRAIN
        if not is_eval_during_training:
            self._increment_step()
        self._prepare_data(data)

        loss, pred, label = self._evaluate_one_step_internal(model, data)

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

        return pred, label, loss

    def _evaluate_one_step_internal(self, model: Module, data: Dict):
        return model(data)

    def _inference(self, model: Module):
        self._init_summary_writer(prefix="inference/worker")
        model.eval()

        with self._get_embedding_writer() as fp:
            with torch.no_grad():
                for data in self.dataset:
                    self._inference_one_step(model, data, fp)
                    if self._should_stop():
                        break

    def _inference_one_step(self, model: Module, data: Dict, fp: Any):
        self._increment_step()
        self._prepare_data(data)

        embedding = self._inference_one_step_internal(model, data)
        self.model.output_embedding(fp, data, embedding)

        if self.args.log_by_steps > 0 and self.step % self.args.log_by_steps == 0:
            duration = self._check_duration()
            self.summary_writer.add_scalar("Inference/Time", duration, self.global_step)
            self.logger.info(
                self._wrap_log(f"step: {self.step:05d}; time: {duration:.4f}s")
            )

    def _inference_one_step_internal(self, model: Module, data: Dict):
        return self.model.get_embedding(data)

    def _increment_step(self):
        self.step += 1
        self.global_step += 1

    def _should_stop(self):
        return self.max_steps > 0 and self.step >= self.max_steps

    def _init_summary_writer(self, prefix: str):
        self.summary_writer = SummaryWriter(
            os.path.join(self.args.metric_dir, f"{prefix}-{self.rank}")
        )

    def _check_duration(self):
        duration = time.time() - self.start_time
        self.start_time = time.time()
        return duration

    def _prepare_data(self, data: Dict):
        if self.args.gpu:
            to_cuda(data)

    def _wrap_log(self, content: str):
        return f"[{self.world_size},{self.rank}] {content}"

    def _create_lr_scheduler(self, optimizer: Optimizer):
        num_training_steps = self.max_steps * self.args.num_epochs
        return (
            get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup * num_training_steps,
                num_training_steps=num_training_steps,
            )
            if self.max_steps > 0
            else None
        )

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

    def _init_max_steps(self):
        self.max_steps = (
            self.args.max_samples // (self.world_size * self.args.batch_size)
            if self.args.max_samples > 0
            else -1
        )
        self.logger.info(self._wrap_log(f"Max steps per epoch:{self.max_steps}"))

    def _get_embedding_writer(self):
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

            return uploader
        return open(embed_path + ".tsv", "w", encoding="utf-8")


class FP16Trainer(Trainer):
    """FP16Trainer supports FP16 mixed precision training with torch.amp or apex."""

    def __init__(self, args: argparse.Namespace):
        """Initialize trainer with command line arguments."""
        assert args.fp16 != FP16_APEX or is_apex_available
        super().__init__(args)

    def _initialize(
        self,
        model: BaseModel,
        dataset: Any,
        optimizer: Optional[Optimizer] = None,
        eval_dataset_for_training: Any = None,
    ):
        model = super()._initialize(
            model, dataset, optimizer, eval_dataset_for_training
        )

        if not self.fp16_enabled():
            return model

        if self.optimizer:
            if self.args.fp16 == FP16_AMP:
                self.grad_scaler = torch.cuda.amp.GradScaler()

            # For training, wrap apex for both model and optimizer.
            if self.args.fp16 == FP16_APEX:
                model, self.optimizer = apex.amp.initialize(
                    model, self.optimizer, opt_level=self.args.apex_opt_level
                )
        else:
            # For evaluation or inference, just wrap apex for model.
            if self.args.fp16 == FP16_APEX:
                model = apex.amp.initialize(model, opt_level=self.args.apex_opt_level)

        return model

    def _apex_backward(self, scaled_loss: torch.Tensor):
        scaled_loss.backward()

    def _apex_step(self):
        self.optimizer.step()

    def _amp_backward(self, loss):
        self.grad_scaler.scale(loss).backward()

    def _amp_step(self):
        self.grad_scaler.step(self.optimizer)

    def _train_one_step_internal(self, model: Module, data: Dict):
        if not self.fp16_enabled():
            return super()._train_one_step_internal(model, data)

        if self.args.fp16 == FP16_APEX:
            loss, score, label = model(data)

            with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                self._apex_backward(scaled_loss)

            if self.args.clip_grad:
                torch.nn.utils.clip_grad_norm_(
                    apex.amp.master_params(self.optimizer), self.args.grad_max_norm
                )

            self._apex_step()

        elif self.args.fp16 == FP16_AMP:
            with torch.cuda.amp.autocast():
                loss, score, label = model(data)

            self._amp_backward(loss)

            if self.args.clip_grad:
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.args.grad_max_norm
                )

            self._amp_step()
            self.grad_scaler.update()
        else:
            raise RuntimeError("Unknown FP16 type.")

        return loss, score, label

    def _evaluate_one_step_internal(self, model: Module, data: Dict):
        if self.args.gpu and self.args.fp16 == FP16_AMP:
            with torch.cuda.amp.autocast():
                return super()._evaluate_one_step_internal(model, data)
        return super()._evaluate_one_step_internal(model, data)

    def _inference_one_step_internal(self, model: Module, data: Dict):
        if self.args.gpu and self.args.fp16 == FP16_AMP:
            with torch.cuda.amp.autocast():
                return super()._inference_one_step_internal(model, data)
        return super()._inference_one_step_internal(model, data)

    def fp16_enabled(self):
        """Check if trainer should use fp16 mode."""
        return self.args.gpu and self.args.fp16 != FP16_NO


class HVDTrainer(FP16Trainer):
    """Horovod based distributed trainer."""

    def __init__(self, args: argparse.Namespace):
        """Initialize horovod."""
        super().__init__(args)
        self._init_hvd()

    def _evaluate(self, model: Module):
        metric, loss = super()._evaluate(model)
        metric = hvd.allreduce(metric)
        loss = hvd.allreduce(loss)
        self.logger.info(
            self._wrap_log(
                f"AllReduced {self.model.metric_name()}: {metric:.4f}; loss: {loss:.4f}"
            )
        )
        return metric, loss

    def _init_hvd(self):
        if self.args.disable_ib:
            disable_infini_band()
        hvd.init()
        self.rank = hvd.rank()
        self.local_rank = hvd.local_rank()
        self.world_size = hvd.size()
        self.logger.info(
            f"Initialized horovod trainer. rank:{self.rank}, local_rank:{self.local_rank},"
            f" world_size:{self.world_size}"
        )

    def _init_model(self, model: BaseModel):
        model = super()._init_model(model)
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        return model

    def _init_optimizer(self, optimizer: Optimizer):
        optimizer = super()._init_optimizer(optimizer)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        compression = (
            hvd.Compression.fp16 if self.fp16_enabled() else hvd.Compression.none
        )
        return hvd.DistributedOptimizer(
            optimizer=optimizer,
            named_parameters=self.model.named_parameters(),
            compression=compression,
            op=hvd.Average,
        )

    def _train_one_epoch(self, model: Module, epoch: int):
        super()._train_one_epoch(model, epoch)
        hvd.join()

    def _inference(self, model: Module):
        super()._inference(model)
        hvd.join()

    def _apex_backward(self, scaled_loss: torch.Tensor):
        scaled_loss.backward()
        self.optimizer.synchronize()

    def _apex_step(self):
        with self.optimizer.skip_synchronize():
            self.optimizer.step()

    def _amp_backward(self, loss):
        self.grad_scaler.scale(loss).backward()
        self.optimizer.synchronize()

    def _amp_step(self):
        with self.optimizer.skip_synchronize():
            self.grad_scaler.step(self.optimizer)


class DDPTrainer(FP16Trainer):
    """Distributed Data Parallel(DDP) based trainer."""

    def __init__(self, args: argparse.Namespace):
        """Initialize trainer from command line arguments."""
        super().__init__(args)
        self._init_process_group()

    def __del__(self):
        """Clear training processes."""
        dist.destroy_process_group()

    def _evaluate(self, model: Module):
        metric, loss = super()._evaluate(model)
        metric = self._allreduce(metric)
        loss = self._allreduce(loss)
        self.logger.info(
            self._wrap_log(
                f"AllReduced {self.model.metric_name()}: {metric:.4f}; loss: {loss:.4f};"
            )
        )
        return metric, loss

    def _init_process_group(self):
        if self.args.disable_ib:
            disable_infini_band()
        # torch.distributed.launch will set below env variables.
        env_dict = {
            key: os.environ[key]
            for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
        }
        self.logger.info(f"Initializing process group with: {env_dict}")
        dist.init_process_group(backend="nccl" if self.args.gpu else "gloo")
        self.rank = dist.get_rank()
        self.local_rank = self.args.local_rank
        self.world_size = dist.get_world_size()

        self.logger.info(
            f"Initialized ddp trainer. rank:{self.rank}, local_rank:{self.local_rank},"
            f" world_size:{self.world_size}"
        )

    def _init_model(self, model: BaseModel):
        model = super()._init_model(model)
        self._broadcast_model_state(model)
        return model

    def _initialize(
        self,
        model: BaseModel,
        dataset: Any,
        optimizer: Optional[Optimizer] = None,
        eval_dataset_for_training: Any = None,
    ):
        model = super()._initialize(
            model, dataset, optimizer, eval_dataset_for_training
        )
        return self._wrap_ddp(model)

    def _broadcast_model_state(self, model: Module):
        vector = parameters_to_vector(model.parameters())
        dist.broadcast(vector, 0)
        if self.rank != 0:
            vector_to_parameters(vector, model.parameters())
        del vector

    def _wrap_ddp(self, model: BaseModel) -> DistributedDataParallel:
        return DistributedDataParallel(  # type: ignore
            model,
            device_ids=[self.local_rank] if self.args.gpu else None,
            output_device=self.local_rank if self.args.gpu else None,
            find_unused_parameters=True,
        )

    def _allreduce(self, metric: torch.Tensor):
        if self.args.gpu:
            metric = metric.cuda()
        dist.all_reduce(metric, op=dist.ReduceOp.SUM)
        return metric / self.world_size


def get_trainer(param):
    """Create trainer from command line arguments."""
    if param.eager:
        # Current TF2 Trainer only use 2.x code.
        # * no v1.Session, placeholder.
        # * use for-loop to access dataset.
        return _get_tf2_trainer(param)
    else:
        # TF1Trainer will disable v2 behavior(`tf.disable_v2_behavior()`),
        # It use `tf.compat.v1` to manage session and dataset, and other 1.x-style functionality.
        # As Tensorflow 2.x support `tf.compat.v1` API, we can still run TF1Trainer in Tensorflow 2.x.
        return _get_tf1_trainer(param)


def _get_tf1_trainer(param):
    """
    Crete a TF trainer.

    Supported:
      * PSTrainer (parameter server)
      * HorovodTFTrainer
    """
    if param.trainer == TrainerType.HVD:
        # Only import hvd_trainer if needed as docker images may not install horovod.
        from deepgnn.tf.common.horovod_trainer import HorovodTFTrainer

        return HorovodTFTrainer(
            trainer=param.trainer,
            model_dir=param.model_dir,
            seed=param.seed,
            user_name=param.user_name,
            job_id=param.job_id,
            gpu=param.gpu,
            log_save_steps=param.log_save_steps,
            summary_save_steps=param.summary_save_steps,
            profiler_save_secs=param.profiler_save_secs,
            checkpoint_save_secs=param.checkpoint_save_secs,
            logger=get_logger(),
        )
    else:
        from deepgnn.tf.common.ps_trainer import PSTrainer

        return PSTrainer(
            trainer=param.trainer,
            model_dir=param.model_dir,
            seed=param.seed,
            ps_hosts=param.ps_hosts,
            job_name=param.job_name,
            worker_hosts=param.worker_hosts,
            task_index=param.task_index,
            user_name=param.user_name,
            job_id=param.job_id,
            gpu=param.gpu,
            log_save_steps=param.log_save_steps,
            summary_save_steps=param.summary_save_steps,
            profiler_save_secs=param.profiler_save_secs,
            checkpoint_save_secs=param.checkpoint_save_secs,
            logger=get_logger(),
        )


def _get_tf2_trainer(param):
    # TODO: support ParameterServerStrategy
    if param.trainer == TrainerType.HVD:
        from deepgnn.tf.common.tf2_horovod_trainer import HorovodEagerTrainer

        return HorovodEagerTrainer(
            trainer=param.trainer,
            model_dir=param.model_dir,
            seed=param.seed,
            user_name=param.user_name,
            job_id=param.job_id,
            gpu=param.gpu,
            log_save_steps=param.log_save_steps,
            summary_save_steps=param.summary_save_steps,
            checkpoint_save_secs=param.checkpoint_save_secs,
            profile_batch=param.profile_batch,
            logger=get_logger(),
        )
    else:
        from deepgnn.tf.common.tf2_trainer import EagerTrainer

        return EagerTrainer(
            model_dir=param.model_dir,
            seed=param.seed,
            user_name=param.user_name,
            job_id=param.job_id,
            gpu=param.gpu,
            log_save_steps=param.log_save_steps,
            summary_save_steps=param.summary_save_steps,
            checkpoint_save_secs=param.checkpoint_save_secs,
            profile_batch=param.profile_batch,
            logger=get_logger(),
        )


class PSTrainer(BaseTFTrainer):
    """TF trainer implementation for PS."""

    def __init__(
        self,
        trainer: TrainerType,
        model_dir: str,
        seed: int,
        ps_hosts: str,
        job_name: str,
        worker_hosts: str,
        task_index: int = 0,
        user_name: str = "",
        job_id: str = "",
        gpu: bool = False,
        log_save_steps: int = 20,
        summary_save_steps: int = 100,
        profiler_save_secs: int = 180,
        checkpoint_save_secs: int = 3600,
        logger: logging.Logger = None,
    ):
        """Initialize trainer."""
        super().__init__(
            model_dir=model_dir,
            seed=seed,
            user_name=user_name,
            job_id=job_id,
            gpu=gpu,
            log_save_steps=log_save_steps,
            summary_save_steps=summary_save_steps,
            profiler_save_secs=profiler_save_secs,
            checkpoint_save_secs=checkpoint_save_secs,
            logger=logger,
        )
        assert trainer == TrainerType.PS

        self.task_index = task_index
        self.ps_hosts = ps_hosts
        self.job_name = job_name
        self.worker_hosts = worker_hosts
        self.worker_size = len(worker_hosts)

        self.tfserver, self.cluster = None, None
        if self.__is_worker() or self.__is_parameter_server():
            # distributed job will set job_name.
            self.tfserver, self.cluster = self.__get_dist_training_server()

        if self.__is_parameter_server():
            # parameter server will join here and never exits.
            tf.compat.v1.logging.info(
                "parameter servier {}-{} starts".format(self.job_name, self.task_index)
            )
            self.__ps_join()

        # MonitoredTrainingSession parameters.
        self.is_chief = self.task_index == 0
        self.session_target = self.tfserver.target if self.tfserver is not None else ""
        self.checkpoint_dir = self.model_dir
        self.config = tf.compat.v1.ConfigProto()
        self.config.gpu_options.allow_growth = True

        self.dist_sync = DistributedSync(
            self.model_dir, self.task_index, len(self.worker_hosts)
        )

        self.parameter_server_num = len(self.ps_hosts)  # type: ignore

    def tf_device(self):
        """Get current device."""
        return tf.compat.v1.device(
            tf.compat.v1.train.replica_device_setter(
                cluster=self.cluster,
                worker_device="/job:worker/task:{}".format(self.task_index),
            )
        )

    def __is_parameter_server(self):
        return self.job_name == "ps"

    def __is_worker(self):
        return self.job_name == "worker"

    def __ps_join(self):
        assert self.job_name == "ps"
        self.tfserver.join()

    def __get_dist_training_server(self):
        assert len(self.ps_hosts) > 0
        assert len(self.worker_hosts) > 0
        clustersepc = tf.train.ClusterSpec(
            {"ps": self.ps_hosts, "worker": self.worker_hosts}
        )
        server = tf.distribute.Server(
            clustersepc, job_name=self.job_name, task_index=self.task_index
        )
        return server, clustersepc


class HorovodTFTrainer(BaseTFTrainer):
    """Distributed training with horovod."""

    def __init__(
        self,
        trainer: TrainerType,
        model_dir: str,
        seed: int,
        user_name: str = "",
        job_id: str = "",
        gpu: bool = False,
        log_save_steps: int = 20,
        summary_save_steps: int = 100,
        profiler_save_secs: int = 180,
        checkpoint_save_secs: int = 3600,
        logger: logging.Logger = None,
    ):
        """Initialize horovod for training."""
        super().__init__(
            model_dir=model_dir,
            seed=seed,
            user_name=user_name,
            job_id=job_id,
            gpu=gpu,
            log_save_steps=log_save_steps,
            summary_save_steps=summary_save_steps,
            profiler_save_secs=profiler_save_secs,
            checkpoint_save_secs=checkpoint_save_secs,
            logger=logger,
        )
        assert trainer == TrainerType.HVD

        hvd.init()
        self.task_index = hvd.rank()
        self.worker_size = hvd.size()
        self.lr_scaler = hvd.size()

        # Hovovod: tf.train.MonitoredTrainingSession: https://github.com/horovod/horovod/blob/master/docs/tensorflow.rst
        # * is_chief: True
        # * master(session_target): ""
        # * checkpoint_dir: accomplish this by passing checkpoint_dir=None to tf.train.MonitoredTrainingSession if hvd.rank() != 0.
        #    - training: DeepGNN use ChiefCheckpointSaverHook, rather than a default CheckpointSaverHook.
        #    - evaluate/inference: DeepGNN set checkpoint_dir=None if hvd.rank() != 0.
        self.checkpoint_dir = (
            self.model_dir if self.task_index == 0 else None  # type: ignore
        )
        if self.gpu:
            self.config = tf.compat.v1.ConfigProto()
            self.config.gpu_options.allow_growth = True
            self.config.gpu_options.visible_device_list = str(hvd.local_rank())
        else:
            self.config = tf.compat.v1.ConfigProto(device_count={"GPU": 0})

        self.dist_sync = DistributedSync(
            self.model_dir, self.task_index, self.worker_size
        )

    def train(
        self,
        dataset: tf.data.Dataset,
        model: tf.keras.Model,
        optimizer: tf.compat.v1.train.Optimizer,
        loss: Union[str, Callable, tf.keras.losses.Loss] = None,
        metrics: List[Union[str, Callable, tf.keras.metrics.Metric]] = None,
        callbacks: List[tf.keras.callbacks.Callback] = None,
        epochs: int = 1,
        steps_per_epoch: int = None,
    ):
        """Wrap the optimizer in hvd.DistributedOptimizer."""
        hvd_optimizer = hvd.DistributedOptimizer(optimizer)
        super().train(
            dataset=dataset,
            model=model,
            loss=loss,
            metrics=metrics,
            optimizer=hvd_optimizer,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
        )

    def inference(
        self,
        dataset: tf.data.Dataset,
        model: tf.keras.Model,
        embedding_to_str_fn: Callable,
        output_embedding_file_prefix: str = "embedding",
        steps: int = None,
    ):
        """Passthrough to the parent."""
        super().inference(
            dataset, model, embedding_to_str_fn, output_embedding_file_prefix, steps
        )

    def evaluate(
        self,
        dataset: tf.data.Dataset,
        model: tf.keras.Model,
        loss: Union[str, Callable, tf.keras.losses.Loss] = None,
        metrics: List[Union[str, Callable, tf.keras.metrics.Metric]] = None,
        _: List[tf.keras.callbacks.Callback] = None,
        steps: int = None,
    ):
        """Passthrough to the parent."""
        super().evaluate(dataset, model, loss=loss, metrics=metrics, steps=steps)

    def _setup_training_hooks(
        self, task_index, checkpoint_dir, global_step, loss, metrics, dist_sync=None
    ):
        """Add hvd.BroadcastGlobalVariablesHook(0) for model variables initliazation."""
        hooks, chiefhooks = super()._setup_training_hooks(
            self.task_index,
            self.checkpoint_dir,
            global_step,
            loss,
            metrics,
            self.dist_sync,
        )
        hooks.append(hvd.BroadcastGlobalVariablesHook(0))
        return hooks, chiefhooks

    def _setup_inference_hooks(self, task_index, global_step, dist_sync=None):
        """Add hvd.BroadcastGlobalVariablesHook(0) for model variables initliazation."""
        hooks = super()._setup_inference_hooks(
            self.task_index, global_step, dist_sync=self.dist_sync
        )
        hooks.append(hvd.BroadcastGlobalVariablesHook(0))
        return hooks

    def _setup_eval_hooks(self, task_index, global_step, loss, metrics, dist_sync=None):
        """Add hvd.BroadcastGlobalVariablesHook(0) for model variables initliazation."""
        hooks = super()._setup_eval_hooks(
            self.task_index, global_step, loss, metrics, self.dist_sync
        )
        hooks.append(hvd.BroadcastGlobalVariablesHook(0))
        return hooks


import argparse
import torch
from typing import Optional, Callable, List
from deepgnn import TrainerType
from deepgnn import get_logger
from contextlib import closing
from deepgnn.pytorch.common import init_common_args
from deepgnn.pytorch.training.args import init_trainer_args, init_fp16_args
from deepgnn.pytorch.training.trainer import Trainer
from deepgnn.graph_engine import create_backend, BackendOptions
from deepgnn.graph_engine.samplers import GENodeSampler, GEEdgeSampler


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

    args = parser.parse_args() if run_args is None else parser.parse_args(run_args)
    for arg in dir(args):
        if not arg.startswith("_"):
            get_logger().info(f"{arg}={getattr(args, arg)}")

    return args


def get_trainer(args: argparse.Namespace) -> Trainer:
    """Create trainer from command line arguments."""
    if args.trainer == TrainerType.BASE:
        return Trainer(args)

    elif args.trainer == TrainerType.DDP:
        from deepgnn.pytorch.training.trainer_ddp import DDPTrainer

        return DDPTrainer(args)
    elif args.trainer == TrainerType.HVD:
        from deepgnn.pytorch.training.trainer_hvd import HVDTrainer

        return HVDTrainer(args)
    else:
        raise RuntimeError(f"Unknown trainer type: {args.trainer}.")


def run_dist(
    init_model_fn: Callable,
    init_dataset_fn: Callable,
    init_optimizer_fn: Optional[Callable] = None,
    init_args_fn: Optional[Callable] = None,
    run_args: Optional[List] = None,
    init_eval_dataset_for_training_fn: Optional[Callable] = None,
):
    """Run distributed training/evaluation/inference.

    Args:
    init_model_fn: (`Callable[args:argparse.Namespace]`)
        Function to initialize gnn model.
    init_dataset_fn: (`Callable[args:argparse.Namespace, graph:Graph, model:BaseModel, rank:int, world_size:int]`)
        Function to initialize dataset.
    init_optimizer_fn: (`Callable[args:argparse.Namespace, model:BaseModel, world_size:int]`, `optional`)
        Function to initialize optimizer, not needed for evaluation/inference.
    init_args_fn: (`Callable[args:argparse.ArgumentParser]`, `optional`)
        Function to add or override command line arguments.
    run_args: (`List[str]`, `optional`)
        List of arguments to pass to argument parser in place of sys.argv, in format ['--data_dir', 'path/to', ...].
    init_eval_dataset_for_training_fn: (`Callable[args:argparse.Namespace, graph:Graph, model:BaseModel, rank:int, world_size:int]`, `optional`)
        Function to initialize evaluation dataset during training.
    """
    args = get_args(init_args_fn, run_args)
    trainer = get_trainer(args)
    backend = create_backend(BackendOptions(args), is_leader=(trainer.rank == 0))

    model = init_model_fn(args)
    dataset = init_dataset_fn(
        args=args,
        model=model,
        rank=trainer.rank,
        world_size=trainer.world_size,
        backend=backend,
    )

    eval_dataloader_for_training = None
    if init_eval_dataset_for_training_fn is not None:
        eval_dataset_for_training = init_eval_dataset_for_training_fn(
            args=args,
            model=model,
            rank=trainer.rank,
            world_size=trainer.world_size,
            backend=backend,
        )
        if eval_dataset_for_training is not None:
            eval_dataloader_for_training = torch.utils.data.DataLoader(
                dataset=eval_dataset_for_training,
                num_workers=args.data_parallel_num,
                prefetch_factor=args.prefetch_factor,
            )
    optimizer = (
        init_optimizer_fn(args=args, model=model, world_size=trainer.world_size)
        if init_optimizer_fn is not None
        else None
    )

    num_workers = (
        0
        if issubclass(dataset.sampler_class, (GENodeSampler, GEEdgeSampler))
        else args.data_parallel_num
    )

    # Executed distributed training/evalution/inference.
    with closing(backend):
        trainer.run(
            model,
            torch.utils.data.DataLoader(
                dataset=dataset,
                num_workers=num_workers,
                prefetch_factor=args.prefetch_factor,
            ),
            optimizer,
            eval_dataloader_for_training,
        )
