"""
New trainer base.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from ray import train
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        layer1 = nn.Linear(input_size, layer_size)
        relu = nn.ReLU()
        layer2 = nn.Linear(layer_size, output_size)

    def forward(self, input):
        return layer2(relu(layer1(input)))


def train_func_distributed():
    num_epochs = 3
    model = NeuralNetwork()
    model = train.torch.prepare_model(model)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(num_epochs):
        output = model(input)
        loss = loss_fn(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"epoch: {epoch}, loss: {loss.item()}")


import argparse
import torch
from typing import Optional, Callable, List
from deepgnn import get_logger
from contextlib import closing
from deepgnn.pytorch.common import init_common_args
from deepgnn.trainer.args import init_trainer_args, init_fp16_args
from deepgnn.graph_engine import create_backend, BackendOptions
from deepgnn.graph_engine.samplers import GENodeSampler, GEEdgeSampler

from deepgnn.pytorch.common.consts import PREFIX_CHECKPOINT, PREFIX_EMBEDDING
from deepgnn.pytorch.common.utils import (
    dump_gpu_memory,
    print_model,
    tally_parameters,
    rotate_checkpoints,
    get_sorted_checkpoints,
    to_cuda,
)
from deepgnn import (
    get_logger,
    log_telemetry,
    TrainMode,
    LOG_PROPS_EVENT_START_WORKER,
    LOG_PROPS_PLATFORM_PYTORCH,
    LOG_PROPS_EVENT_END_WORKER,
)
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from deepgnn.pytorch.common.optimization import get_linear_schedule_with_warmup


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

    args = parser.parse_known_args()[0] if run_args is None else parser.parse_args(run_args)
    for arg in dir(args):
        if not arg.startswith("_"):
            get_logger().info(f"{arg}={getattr(args, arg)}")

    return args


def _train_loop(
    config: dict
):
    def _save_checkpoint(epoch: int):
        # Don't save for last step to avoid duplication with ckpt after epoch finished.
        if max_steps > 0 and step == max_steps:
            return

        save_path = os.path.join(
            f"{args.save_path}",
            f"{PREFIX_CHECKPOINT}-{epoch:03}-{step:06}.pt",
        )
        torch.save(
            {"state_dict": model.state_dict(), "epoch": epoch, "step": step},
            save_path,
        )
        logger.info(_wrap_log(f"Saved checkpoint to {save_path}."))
        rotate_checkpoints(args.save_path, args.max_saved_ckpts)
    def _wrap_log(content: str):
        return f"[{world_size},{rank}] {content}"

    def _load_checkpoint(ckpt_path: str = None):
        if not ckpt_path:
            # Search and sort checkpoints from model path.
            ckpts = get_sorted_checkpoints(args.model_dir)
            ckpt_path = ckpts[-1] if len(ckpts) > 0 else None

        if ckpt_path is not None:

            init_ckpt = torch.load(ckpt_path, map_location="cpu")
            epochs_trained = init_ckpt["epoch"]
            steps_in_epoch_trained = init_ckpt["step"]

            # Only worker with rank 0 loads state dict.
            if rank == 0:
                model.load_state_dict(init_ckpt["state_dict"])
                logger.info(
                    _wrap_log(
                        f"Loaded initial checkpoint: {ckpt_path},"
                        f" trained epochs: {epochs_trained}, steps: {steps_in_epoch_trained}"
                    )
                )

            del init_ckpt

    local_rank = 0
    steps_in_epoch_trained = 0
    rank = 0
    world_size = 1
    max_steps = -1


    init_model_fn = config["init_model_fn"]
    init_dataset_fn = config["init_dataset_fn"]
    init_optimizer_fn = config["init_optimizer_fn"]
    init_args_fn = config["init_args_fn"]
    run_args = config["run_args"]
    init_eval_dataset_for_training_fn = config["init_eval_dataset_for_training_fn"]

    args = get_args(init_args_fn, run_args)

    model = init_model_fn(args)
    model = train.torch.prepare_model(model)
    model.train()

    if args.gpu:
        torch.cuda.set_device(local_rank)
        model.cuda()

    if rank == 0:
        if (
            not args.enable_adl_uploader
            or args.mode != TrainMode.INFERENCE
        ):
            os.makedirs(args.save_path, exist_ok=True)
        print_model(model)
        tally_parameters(model)

    _load_checkpoint()

    backend = create_backend(
        BackendOptions(args), is_leader=(rank == 0)
    )

    dataset = init_dataset_fn(
        args=args,
        model=model,
        rank=rank,
        world_size=world_size,
        backend=backend,
    )

    optimizer = (
        init_optimizer_fn(args=args, model=model, world_size=world_size)
        if init_optimizer_fn is not None
        else None
    )
    if optimizer:
        num_training_steps = max_steps * args.num_epochs
        lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup * num_training_steps,
                num_training_steps=num_training_steps,
            ) if max_steps > 0 else None

    num_workers = (
        0
        if issubclass(dataset.sampler_class, (GENodeSampler, GEEdgeSampler))
        else args.data_parallel_num
    )

    logger=get_logger()


    model_name = type(model).__name__

    max_steps = (
        args.max_samples // (world_size * args.batch_size)
        if args.max_samples > 0
        else -1
    )
    logger.info(_wrap_log(f"Max steps per epoch:{max_steps}"))

    # On-demand enable telemetry.
    log_telemetry(
        logger,
        f"Training worker started. Model: {model_name}.",
        LOG_PROPS_EVENT_START_WORKER,
        f"{args.mode}",
        model_name,
        args.user_name,
        args.job_id,
        rank,
        world_size,
        LOG_PROPS_PLATFORM_PYTORCH,
    )

    result = None

    epochs_trained = 0
    with closing(backend):
        start_time = time.time()
        summary_writer = SummaryWriter(
            os.path.join(args.metric_dir, f"train/worker-{rank}")
        )
        model.train()
        lr_scheduler = None

        # Continous training from epoch and step saved in checkpoint.
        step = steps_in_epoch_trained
        global_step = step
        for epoch in range(epochs_trained, args.num_epochs):
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                num_workers=num_workers,
                prefetch_factor=args.prefetch_factor,
            )
            for i, data in enumerate(dataloader):
                # Skip trained steps.
                if i < step:
                    continue

                train_losses = []  # type: List[float]
                train_metrics = []  # type: List[float]

                step += 1
                global_step += 1
                if args.gpu:
                    to_cuda(data)

                optimizer.zero_grad()

                loss, pred, label = model(data)
                loss.backward()
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_max_norm)
                optimizer.step()

                train_losses.append(loss.data.item())

                if lr_scheduler:
                    lr_scheduler.step()

                if (
                    rank == 0
                    and args.save_ckpt_by_steps > 0
                    and step % args.save_ckpt_by_steps == 0
                ):
                    _save_checkpoint(epoch)

                # Calculate training metrics for one batch data.
                metric = (
                    model.compute_metric([pred], [label]).data.item()
                    if args.use_per_step_metrics
                    else torch.tensor(0.0)
                )
                train_metrics.append(metric)

                if args.log_by_steps > 0 and step % args.log_by_steps == 0:
                    train_loss = np.mean(train_losses)
                    train_metric = np.mean(train_metrics)
                    train_losses = []
                    train_metrics = []
                    duration = time.time() - start_time
                    start_time = time.time()
                    summary_writer.add_scalar(
                        "Training/Loss", train_loss, global_step
                    )
                    summary_writer.add_scalar("Training/Time", duration, global_step)
                    if args.use_per_step_metrics:
                        summary_writer.add_scalar(
                            f"Training/{model.metric_name()}",
                            train_metric,
                            global_step,
                        )
                    if lr_scheduler:
                        summary_writer.add_scalar(
                            "Training/LR", lr_scheduler.get_last_lr()[0], global_step
                        )

                    logger.info(
                        _wrap_log(
                            f"epoch: {epoch}; step: {step:05d}; loss: {train_loss:.4f};"
                            + (
                                f" {model.metric_name()}: {train_metric:.4f}; time: {duration:.4f}s"
                                if args.use_per_step_metrics
                                else f" time: {duration:.4f}s"
                            )
                        )
                    )

                '''
                if (
                    eval_dataset_for_training is not None
                    and eval_during_train_by_steps > 0
                    and step % eval_during_train_by_steps == 0
                ):
                    model.eval()
                    eval_metric, eval_loss = _evaluate(model)
                    if args.use_per_step_metrics:
                        summary_writer.add_scalar(
                            f"Validation/{model.metric_name()}",
                            eval_metric,
                            global_step,
                        )
                        summary_writer.add_scalar(
                            "Validation/Loss", eval_loss, global_step
                        )
                    logger.info(
                        _wrap_log(
                            f"epoch: {epoch}; step: {step:05d};"
                            + (
                                f" Validation/{model.metric_name()}: {eval_metric:.4f}, Validation/Loss: {eval_loss:.4f}"
                                if args.use_per_step_metrics
                                else ""
                            )
                        )
                    )
                    model.train()
                '''

                if max_steps > 0 and step >= max_steps:
                    break


            # Epoch finishes.
            step = 0
            if rank == 0 and epoch % args.save_ckpt_by_epochs == 0:
                _save_checkpoint(epoch + 1)





        log_telemetry(
            logger,
            f"Training worker finished. Model: {model_name}.",
            LOG_PROPS_EVENT_END_WORKER,
            f"{args.mode}",
            model_name,
            args.user_name,
            args.job_id,
            rank,
            world_size,
            LOG_PROPS_PLATFORM_PYTORCH,
        )

        if args.gpu:
            logger.info(_wrap_log(dump_gpu_memory()))

        return result



import deepgnn.graph_engine.snark._lib as lib

import platform
import os
def get_lib_name():
    lib_name = "libwrapper.so"
    if platform.system() == "Windows":
        lib_name = "wrapper.dll"

    _SNARK_LIB_PATH_ENV_KEY = "SNARK_LIB_PATH"
    #if _SNARK_LIB_PATH_ENV_KEY in os.environ:
    #    return os.environ[_SNARK_LIB_PATH_ENV_KEY]

    os.environ[_SNARK_LIB_PATH_ENV_KEY] = os.path.join("/home/user/DeepGNN/bazel-bin/src/cc/lib", lib_name)
    return os.environ[_SNARK_LIB_PATH_ENV_KEY]


def setup_module():
    lib._LIB_PATH = get_lib_name()

def run_dist(
    init_model_fn: Callable,
    init_dataset_fn: Callable,
    init_optimizer_fn: Optional[Callable] = None,
    init_args_fn: Optional[Callable] = None,
    run_args: Optional[List] = None,
    init_eval_dataset_for_training_fn: Optional[Callable] = None,
):
    setup_module()
    import ray
    ray.init(num_cpus=2, num_gpus=0)
    try:
        trainer = TorchTrainer(
            _train_loop,
            train_loop_config={
                "init_model_fn": init_model_fn,
                "init_dataset_fn": init_dataset_fn,
                "init_optimizer_fn": init_optimizer_fn,
                "init_args_fn": init_args_fn,
                "run_args": run_args,
                "init_eval_dataset_for_training_fn": init_eval_dataset_for_training_fn,
            },
            scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
        )

        results = trainer.fit()
    except Exception as e:
        ray.shutdown()
        raise e
    ray.shutdown()
