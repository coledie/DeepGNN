# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import sys
import os
import platform
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Sampler

import ray
import ray.train as train
from ray.train.torch import TorchTrainer
from ray.air import session
from ray.air.config import ScalingConfig

from deepgnn.graph_engine import SamplingStrategy
from deepgnn.graph_engine.snark.local import Client


import pandas as pd

feature_idx = 1
feature_dim = 50
label_idx = 0
label_dim = 121


def setup_module(module):
    import deepgnn.graph_engine.snark._lib as lib

    lib_name = "libwrapper.so"
    if platform.system() == "Windows":
        lib_name = "wrapper.dll"

    os.environ[lib._SNARK_LIB_PATH_ENV_KEY] = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "src", "cc", "lib", lib_name
    )


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def query(self, g, idx):
        return {
            "features": g.node_features(
                idx, np.array([[feature_idx, feature_dim]]), feature_type=np.float32
            ),
            "labels": np.ones((len(idx))),
        }


def train_func(config: Dict):
    worker_batch_size = config["batch_size"] // session.get_world_size()

    model = NeuralNetwork()
    model = train.torch.prepare_model(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
    optimizer = train.torch.prepare_optimizer(optimizer)

    loss_fn = nn.CrossEntropyLoss()

    loss_results = []

    dataset = ray.data.range(2708, parallelism=1)
    pipe = dataset.window(blocks_per_window=2)
    g = Client("/tmp/cora", [0], delayed_start=True)

    def transform_batch(batch: list) -> dict:
        return NeuralNetwork.query(None, g, batch)

    pipe = pipe.map_batches(transform_batch)

    model.train()
    for train_dataloader in pipe.repeat(config["epochs"]).iter_epochs():
        for i, batch in enumerate(
            train_dataloader.random_shuffle_each_window().iter_torch_batches(
                batch_size=worker_batch_size
            )
        ):
            pred = model(batch["features"])
            loss = loss_fn(pred, batch["labels"].squeeze().long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss_results


def test_graphsage_ppi_hvd_trainer():
    ray.init()
    trainer = TorchTrainer(
        train_func,
        train_loop_config={"lr": 1e-3, "batch_size": 64, "epochs": 4},
        scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
    )
    result = trainer.fit()
    print(f"Results: {result.metrics}")


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
