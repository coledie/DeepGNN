# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import tempfile
import time
import numpy.testing as npt
import os
import sys
import torch
import argparse
import numpy as np
import urllib.request
import zipfile

from deepgnn import get_logger
from deepgnn.pytorch.common import MRR, F1Score
from deepgnn.pytorch.common.dataset import TorchDeepGNNDataset
from deepgnn.pytorch.encoding.feature_encoder import (
    TwinBERTEncoder,
    TwinBERTFeatureEncoder,
)
from examples.pytorch.conftest import (  # noqa: F401
    MockSimpleDataLoader,
    MockFixedSimpleDataLoader,
    mock_graph,
)
from deepgnn.graph_engine import (
    FeatureType,
    GraphType,
    BackendType,
    BackendOptions,
    GENodeSampler,
    create_backend,
)
import deepgnn.graph_engine.snark.convert as convert
from deepgnn.graph_engine.snark.decoders import JsonDecoder
from deepgnn.graph_engine.snark.converter.options import DataConverterType
from model import SupervisedGraphSage, UnSupervisedGraphSage  # type: ignore
import deepgnn.graph_engine.snark._lib as lib
import platform
import os

logger = get_logger()


def setup_module(module):
    lib_name = "libwrapper.so"
    if platform.system() == "Windows":
        lib_name = "wrapper.dll"

    os.environ[lib._SNARK_LIB_PATH_ENV_KEY] = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "src", "cc", "lib", lib_name
    )


from main import init_args, create_model, create_dataset, create_optimizer  # type: ignore
from deepgnn.graph_engine.trainer.factory import run_dist
import deepgnn.graph_engine.snark.local as local


def test_graphsage_ppi_ddp_amp_trainer(mock_graph):
    torch.manual_seed(0)
    np.random.seed(0)
    data_dir = "/tmp/ppi"
    num_nodes = 56944
    num_classes = 121
    label_dim = 121
    label_idx = 1
    feature_dim = 50
    feature_idx = 0
    edge_type = 0

    model_path = tempfile.TemporaryDirectory()
    model_path_name = model_path.name + "/gnnmodel.pt"

    run_args = f"""--data_dir {data_dir} --mode train --trainer ddp --train_workers 2 --seed 123 \
--backend snark --graph_type local --converter skip \
--batch_size 140 --learning_rate 0.0001 --num_epochs 4 \
--node_type 0 --max_id -1 \
--model_dir {model_path.name} --metric_dir {model_path.name} --save_path {model_path.name} \
--feature_idx {feature_idx} --feature_dim {feature_dim} --label_idx {label_idx} --label_dim {label_dim} --algo supervised \
--log_by_steps 1 --use_per_step_metrics --fp16 amp""".split()

    run_dist(
        init_model_fn=create_model,
        init_dataset_fn=create_dataset,
        init_optimizer_fn=create_optimizer,
        init_args_fn=init_args,
        run_args=run_args,
    )

    metric = F1Score()
    g = local.Client(data_dir, [0, 1])  # mock_graph
    graphsage = SupervisedGraphSage(
        num_classes=num_classes,
        metric=F1Score(),
        label_idx=label_idx,
        label_dim=label_dim,
        feature_type=FeatureType.FLOAT,
        feature_idx=feature_idx,
        feature_dim=feature_dim,
        edge_type=edge_type,
        fanouts=[5, 5],
    )

    graphsage.load_state_dict(
        {
            key.replace("module.", ""): value
            for key, value in torch.load(model_path_name)["state_dict"].items()
        }
    )
    graphsage.eval()

    # Generate validation dataset from random indices
    rand_indices = np.random.RandomState(seed=1).permutation(num_nodes)
    val_ref = rand_indices[1000:1500]
    simpler = MockFixedSimpleDataLoader(val_ref, query_fn=graphsage.query, graph=g)
    trainloader = torch.utils.data.DataLoader(simpler)
    it = iter(trainloader)
    loss, pred, label = graphsage(it.next())

    # val_output_ref = graphsage.get_score(it.next())
    # val_labels = g.node_features(
    #    val_ref, np.array([[label_idx, label_dim]]), FeatureType.FLOAT
    # ).argmax(1)
    f1_ref = graphsage.compute_metric([pred], [label])

    assert 0.5 < f1_ref and f1_ref < 0.6
    model_path.cleanup()


def test_graphsage_ppi_hvd_amp_trainer(mock_graph):
    torch.manual_seed(0)
    np.random.seed(0)
    data_dir = "/tmp/ppi"
    num_nodes = 56944
    num_classes = 121
    label_dim = 121
    label_idx = 1
    feature_dim = 50
    feature_idx = 0
    edge_type = 0

    model_path = tempfile.TemporaryDirectory()
    model_path_name = model_path.name + "/gnnmodel.pt"

    run_args = f"""--data_dir {data_dir} --mode train --trainer hvd --train_workers 2 --seed 123 \
--backend snark --graph_type local --converter skip \
--batch_size 140 --learning_rate 0.0001 --num_epochs 4 \
--node_type 0 --max_id -1 \
--model_dir {model_path.name} --metric_dir {model_path.name} --save_path {model_path.name} \
--feature_idx {feature_idx} --feature_dim {feature_dim} --label_idx {label_idx} --label_dim {label_dim} --algo supervised \
--log_by_steps 1 --use_per_step_metrics --fp16 amp""".split()

    run_dist(
        init_model_fn=create_model,
        init_dataset_fn=create_dataset,
        init_optimizer_fn=create_optimizer,
        init_args_fn=init_args,
        run_args=run_args,
    )

    metric = F1Score()
    g = local.Client(data_dir, [0, 1])  # mock_graph
    graphsage = SupervisedGraphSage(
        num_classes=num_classes,
        metric=F1Score(),
        label_idx=label_idx,
        label_dim=label_dim,
        feature_type=FeatureType.FLOAT,
        feature_idx=feature_idx,
        feature_dim=feature_dim,
        edge_type=edge_type,
        fanouts=[5, 5],
    )

    graphsage.load_state_dict(
        {
            key.replace("module.", ""): value
            for key, value in torch.load(model_path_name)["state_dict"].items()
        }
    )
    graphsage.eval()

    # Generate validation dataset from random indices
    rand_indices = np.random.RandomState(seed=1).permutation(num_nodes)
    val_ref = rand_indices[1000:1500]
    simpler = MockFixedSimpleDataLoader(val_ref, query_fn=graphsage.query, graph=g)
    trainloader = torch.utils.data.DataLoader(simpler)
    it = iter(trainloader)
    loss, pred, label = graphsage(it.next())

    # val_output_ref = graphsage.get_score(it.next())
    # val_labels = g.node_features(
    #    val_ref, np.array([[label_idx, label_dim]]), FeatureType.FLOAT
    # ).argmax(1)
    f1_ref = graphsage.compute_metric([pred], [label])

    assert 0.5 < f1_ref and f1_ref < 0.6
    model_path.cleanup()


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )
