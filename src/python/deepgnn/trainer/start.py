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
        self.layer1 = nn.Linear(input_size, layer_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(layer_size, output_size)

    def forward(self, input):
        return self.layer2(self.relu(self.layer1(input)))


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


def _train_loop(
    config: dict
):
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

    backend = create_backend(
        BackendOptions(args), is_leader=(0 == 0)
    )  # TODO is_leader

    dataset = init_dataset_fn(
        args=args,
        model=model,
        rank=0,#trainer.rank,
        world_size=1,#trainer.world_size,
        backend=backend,
    )

    optimizer = (
        init_optimizer_fn(args=args, model=model, world_size=1)#trainer.world_size)
        if init_optimizer_fn is not None
        else None
    )

    num_workers = (
        0
        if issubclass(dataset.sampler_class, (GENodeSampler, GEEdgeSampler))
        else args.data_parallel_num
    )

    # TODO torch trainer _initialiez

    # Executed distributed training/evalution/inference.
    with closing(backend):
        """
        trainer.run(
            model,
            ,
            optimizer,
            eval_dataloader_for_training,
        )
        """
        d = torch.utils.data.DataLoader(
                dataset=dataset,
                num_workers=num_workers,
                prefetch_factor=args.prefetch_factor,
            )
        for epoch, data in enumerate(d):
            loss, pred, label = model(data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"epoch: {epoch}, loss: {loss.item()}")


def run_dist(
    init_model_fn: Callable,
    init_dataset_fn: Callable,
    init_optimizer_fn: Optional[Callable] = None,
    init_args_fn: Optional[Callable] = None,
    run_args: Optional[List] = None,
    init_eval_dataset_for_training_fn: Optional[Callable] = None,
):
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


if __name__ == "__main__":
    num_samples = 20
    input_size = 10
    layer_size = 15
    output_size = 5

    # In this example we use a randomly generated dataset.
    input = torch.randn(num_samples, input_size)
    labels = torch.randn(num_samples, output_size)

    # For GPU Training, set `use_gpu` to True.
    use_gpu = False

    trainer = TorchTrainer(
        train_func_distributed,
        scaling_config=ScalingConfig(num_workers=4, use_gpu=use_gpu),
    )

    results = trainer.fit()
