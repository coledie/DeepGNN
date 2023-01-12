**************************************************
Ray Usage Example for Node Classification with GAT
**************************************************

In this guide we use a pre-built `Graph Attention Network(GAT) <https://arxiv.org/abs/1710.10903>`_ model to classify nodes in the `Cora dataset <https://graphsandnetworks.com/the-cora-dataset/>`_. Readers can expect an understanding of the DeepGNN experiment flow and details on model design.

Cora Dataset
============
The Cora dataset consists of 2708 scientific publications represented as nodes interconnected by 5429 reference links represented as edges. Each paper is described by a binary mask for 1433 pertinent dictionary words and an integer in {0..6} representing its type.
First we download the Cora dataset and convert it to a valid binary representation via our built-in Cora downloader.

.. code-block:: python

    >>> import tempfile
	>>> from deepgnn.graph_engine.data.citation import Cora
    >>> data_dir = tempfile.TemporaryDirectory()
	>>> Cora(data_dir.name)
	<deepgnn.graph_engine.data.citation.Cora object at 0x...>

GAT Model
=========

Using this Graph Attention Network, we can accurately predict which category a specific paper belongs to based on its dictionary and the dictionaries of papers it references.
This model leverages masked self-attentional layers to address the shortcomings of graph convolution based models. By stacking layers in which nodes are able to attend over their neighborhoods features, we enable the model to specify different weights to different nodes in a neighborhood, without requiring any kind of costly matrix operation (such as inversion) or the knowledge of the graph structure up front.

`Paper <https://arxiv.org/abs/1710.10903>`_, `author's code <https://github.com/PetarV-/GAT>`_.

Next we copy the GAT model from `DeepGNN's examples directory <https://github.com/microsoft/DeepGNN/blob/main/examples/tensorflow/gat>`_. Pre-built models are kept out of the pip installation because it is rarely possible to inheret and selectively edit a single function of a graph model, instead it is best to copy the entire model and edit as needed.
DeepGNN models typically contain multiple parts:

	1. Query struct and implementation
	2. Model init and forward
	3. Training setup: Dataset, Optimizer, Model creation
	4. Execution

Setup
======

Combined imports from `model.py <https://github.com/microsoft/DeepGNN/blob/main/examples/tensorflow/gat/gat.py>`_ and `main.py <https://github.com/microsoft/DeepGNN/blob/main/examples/tensorflow/gat/main.py>`_.

.. code-block:: python

    >>> import tempfile
    >>> import os
    >>> import numpy as np
    >>> import tensorflow as tf
    >>> from dataclasses import dataclass
    >>> from typing import Dict, List
    >>> import ray
    >>> from ray.train.tensorflow import TensorflowTrainer
    >>> from ray.air import session
    >>> from ray.air.config import ScalingConfig, RunConfig
    >>> from deepgnn.graph_engine import Graph, graph_ops
    >>> from deepgnn.graph_engine import (
    ...    SamplingStrategy,
    ...    GENodeSampler,
    ... )
    >>> from deepgnn.graph_engine.snark.local import Client
    >>> from deepgnn.tf.nn.gat_conv import GATConv
    >>> from deepgnn.tf.nn.metrics import masked_accuracy, masked_softmax_cross_entropy
    >>> from deepgnn.tf.common.dataset import create_tf_dataset

Query
=====
Query is the interface between the model and graph engine. It is used by the trainer to fetch contexts which will be passed as input to the model forward function. Since query is a separate function, the trainer may pre-fetch contexts allowing graph engine operations and model training to occur in parallel.
In the GAT model, query samples neighbors repeatedly `num_hops` times in order to generate a sub-graph. All node and edge features in this sub-graph are pulled and added to the context.

.. code-block:: python

    >>> @dataclass
    ... class GATQueryParameter:
    ...    neighbor_edge_types: np.array
    ...    feature_idx: int
    ...    feature_dim: int
    ...    label_idx: int
    ...    label_dim: int
    ...    feature_type: np.dtype = np.float32
    ...    label_type: np.dtype = np.float32
    ...    num_hops: int = 2

    >>> class GATQuery:
    ...    """Graph Query: get sub graph for GAT training"""
    ...
    ...    def __init__(self, param: GATQueryParameter):
    ...        self.param = param
    ...        self.label_meta = np.array([[param.label_idx, param.label_dim]], np.int32)
    ...        self.feat_meta = np.array([[param.feature_idx, param.feature_dim]], np.int32)
    ...
    ...    def query_training(
    ...        self, graph: Graph, inputs: np.array, return_shape: bool = False
    ...    ):
    ...        nodes, edges, src_idx = graph_ops.sub_graph(
    ...            graph=graph,
    ...            src_nodes=inputs,
    ...            edge_types=self.param.neighbor_edge_types,
    ...            num_hops=self.param.num_hops,
    ...            self_loop=True,
    ...            undirected=True,
    ...            return_edges=True,
    ...        )
    ...        input_mask = np.zeros(nodes.size, np.bool_)
    ...        input_mask[src_idx] = True
    ...
    ...        feat = graph.node_features(nodes, self.feat_meta, self.param.feature_type)
    ...        label = graph.node_features(nodes, self.label_meta, self.param.label_type)
    ...        label = label.astype(np.int32)
    ...
    ...        edges_value = np.ones(edges.shape[0], np.float32)
    ...        adj_shape = np.array([nodes.size, nodes.size], np.int64)
    ...        graph_tensor = (nodes, feat, input_mask, label, edges, edges_value, adj_shape)
    ...        if return_shape:
    ...            # fmt: off
    ...            # N is the number of `nodes`, which is variable because `inputs` nodes are different.
    ...            N = None
    ...            shapes = (
    ...                [N],                            # Nodes
    ...                [N, self.param.feature_dim],    # feat
    ...                [N],                            # input_mask
    ...                [N, self.param.label_dim],      # label
    ...                [None, 2],                      # edges
    ...                [None],                         # edges_value
    ...                [2]                             # adj_shape
    ...            )
    ...            # fmt: on
    ...            return graph_tensor, shapes
    ...
    ...        return graph_tensor

Model Forward and Init
======================
The model init and forward functions look the same as any other tensorflow model. The call function is expected to return three values: the model predictions for given nodes, the batch loss and metrics.

.. code-block:: python

    >>> class GAT(tf.keras.Model):
    ...    """ GAT Model (supervised)"""
    ...
    ...    def __init__(
    ...        self,
    ...        head_num: List[int] = [8, 1],
    ...        hidden_dim: int = 8,
    ...        num_classes: int = -1,
    ...        ffd_drop: float = 0.0,
    ...        attn_drop: float = 0.0,
    ...        l2_coef: float = 0.0005,
    ...    ):
    ...        super().__init__()
    ...        self.num_classes = num_classes
    ...        self.l2_coef = l2_coef
    ...
    ...        self.out_dim = num_classes
    ...
    ...        self.input_layer = GATConv(
    ...            attn_heads=head_num[0],
    ...            out_dim=hidden_dim,
    ...            act=tf.nn.elu,
    ...            in_drop=ffd_drop,
    ...            coef_drop=attn_drop,
    ...            attn_aggregate="concat",
    ...        )
    ...        ## TODO: support hidden layer
    ...        assert len(head_num) == 2
    ...        self.out_layer = GATConv(
    ...            attn_heads=head_num[1],
    ...            out_dim=self.out_dim,
    ...            act=None,
    ...            in_drop=ffd_drop,
    ...            coef_drop=attn_drop,
    ...            attn_aggregate="average",
    ...        )
    ...
    ...    def forward(self, feat, bias_mat, training):
    ...        h_1 = self.input_layer([feat, bias_mat], training=training)
    ...        out = self.out_layer([h_1, bias_mat], training=training)
    ...        return out
    ...
    ...    def call(self, inputs, training=True):
    ...        # inputs: nodes    feat      mask    labels   edges       edges_value  adj_shape
    ...        # shape:  [N]      [N, F]    [N]     [N]      [num_e, 2]  [num_e]      [2]
    ...        nodes, feat, mask, labels, edges, edges_value, adj_shape = inputs
    ...
    ...        # bias_mat = -1e9 * (1.0 - adj)
    ...        sp_adj = tf.SparseTensor(edges, edges_value, adj_shape)
    ...        logits = self.forward(feat, sp_adj, training)
    ...
    ...        ## embedding results
    ...        self.src_emb = tf.boolean_mask(logits, mask)
    ...        self.src_nodes = tf.boolean_mask(nodes, mask)
    ...
    ...        labels = tf.one_hot(labels, self.num_classes)
    ...        logits = tf.reshape(logits, [-1, self.num_classes])
    ...        labels = tf.reshape(labels, [-1, self.num_classes])
    ...        mask = tf.reshape(mask, [-1])
    ...
    ...        ## loss
    ...        xent_loss = masked_softmax_cross_entropy(logits, labels, mask)
    ...        loss = xent_loss + self.l2_loss()
    ...
    ...        ## metric
    ...        acc = masked_accuracy(logits, labels, mask)
    ...        return logits, loss, {"accuracy": acc}
    ...
    ...    def l2_loss(self):
    ...        vs = []
    ...        for v in self.trainable_variables:
    ...            vs.append(tf.nn.l2_loss(v))
    ...        lossL2 = tf.add_n(vs) * self.l2_coef
    ...        return lossL2
    ...
    ...    def train_step(self, data: dict):
    ...        """override base train_step."""
    ...        with tf.GradientTape() as tape:
    ...            _, loss, metrics = self(data, training=True)
    ...
    ...        grads = tape.gradient(loss, self.trainable_variables)
    ...        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
    ...        result = {"loss": loss, **metrics}
    ...        result.update(metrics)
    ...        return result

Model Init
==========
We need to implement the `build_model` function to allow distributed workers initialize the model.

.. code-block:: python

    >>> def build_model():
    ...    p = GATQueryParameter(
    ...        neighbor_edge_types=np.array([0], np.int32),
    ...        feature_idx=0,
    ...        feature_dim=1433,
    ...        label_idx=1,
    ...        label_dim=1,
    ...        num_hops=len([8, 1]),
    ...    )
    ...    query_obj = GATQuery(p)
    ...
    ...    model = GAT(
    ...        head_num=[8, 1],
    ...        hidden_dim=8,
    ...        num_classes=7,
    ...        ffd_drop=.6,
    ...        attn_drop=.6,
    ...        l2_coef=0.0005,
    ...    )
    ...
    ...    return model, query_obj

Ray Train
=========

Here we define our training function.
Then we define a standard tf training loop using the ray dataset, with no changes to model or optimizer usage.

.. code-block:: python

    >>> def train_func(config: Dict):
    ...     tf.keras.utils.set_random_seed(0)
    ...
    ...     model, query = build_model()
    ...
    ...     tf_dataset, steps_per_epoch = create_tf_dataset(
    ...         sampler_class=GENodeSampler,
    ...         query_fn=query.query_training,
    ...         backend=type("Backend", (object,), {"graph": Client(config["data_dir"], [0])})(),
    ...         node_types=np.array([0], dtype=np.int32),
    ...         batch_size=config["batch_size"],
    ...         num_workers=2,
    ...         worker_index=0,
    ...         strategy=SamplingStrategy.RandomWithoutReplacement,
    ...     )
    ...
    ...     model.optimizer = tf.keras.optimizers.Adam(
    ...         learning_rate=.005
    ...     )
    ...
    ...     with tf.distribute.get_strategy().scope():
    ...         model.compile(optimizer=model.optimizer)
    ...
    ...     for epoch in range(config["n_epochs"]):
    ...         history = model.fit(tf_dataset, verbose=0)
    ...         session.report(history.history)

In this step we start the training job.
First we start a local ray cluster with `ray.init() <https://docs.ray.io/en/latest/ray-core/package-ref.html#ray-init>`_.
Next we initialize a `TensorflowTrainer <https://docs.ray.io/en/latest/ray-air/package-ref.html#tensorflow>`_
object to wrap our training loop. This takes parameters that go to the training loop and parameters
to define number workers and cpus/gpus used.
Finally we call trainer.fit() to execute the training loop.

.. code-block:: python

    >>> ray.init(num_cpus=3)
    RayContext(...)

    >>> trainer = TensorflowTrainer(
    ...     train_loop_per_worker=train_func,
    ...     train_loop_config={
    ...         "batch_size": 2708,
    ...         "data_dir": data_dir.name,
    ...         "n_epochs": 100,
    ...     },
    ...     run_config=RunConfig(verbose=0),
    ...     scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
    ... )
    >>> result = trainer.fit()
    >>> result.metrics["accuracy"]
    [0.8...]
