# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Reddit dataset."""

import argparse
import json
import os
import zipfile
from typing import List, Dict

import numpy as np
from sklearn.preprocessing import StandardScaler
import deepgnn.graph_engine.snark.convert as convert
import deepgnn.graph_engine.snark.decoders as decoders

from deepgnn.graph_engine.data.data_util import download_file
from deepgnn.graph_engine.snark.local import Client


def onehot(value, size):
    """Convert value to onehot vector."""
    output = [0] * size
    output[value] = 1
    return output


class Reddit(Client):
    """
    Reddit Subreddit-Subreddit Interactions graph.

    Sampled 50 large communities and built a post-to-post graph, connecting posts if the same user comments
    on both. For features, we use off-the-shelf 300-dimensional GloVe CommonCrawl word vectors.

    References:
        http://snap.stanford.edu/graphsage/#datasets
        https://github.com/williamleif/GraphSAGE/blob/master/graphsage/utils.py

    Graph Statistics:
    - Nodes: 232,965
    - Edges: 114,618,780 * edge_downsample_pct
    - Split: (Train: 152410, Valid: 23699, Test: 55334)
    - Node Label Dim: 50 (id:0)
    - Node Feature Dim: 300 (id:1)
    """

    def __init__(self, output_dir: str = None, edge_downsample_pct: float = 0.1):
        """
        Initialize Reddit dataset.

        Args:
          output_dir (string): file directory for graph data.
          edge_downsample_pct (float, default=.1): Percent of edges to use, default reddit graph has 100M edges.
        """
        self._edge_downsample_pct = edge_downsample_pct
        self.url = "https://snap.stanford.edu/graphsage/reddit.zip"
        self.GRAPH_NAME = "reddit"
        self.output_dir = output_dir
        if self.output_dir is None:
            self.output_dir = os.path.join("/tmp/", self.GRAPH_NAME)
        self._build_graph(self.output_dir)
        super().__init__(path=self.output_dir, partitions=[0])

    def data_dir(self):
        """Graph location on disk."""
        return self.output_dir

    def _load_raw_graph(self, data_dir: str):
        id_map = json.load(open(os.path.join(data_dir, "reddit-id_map.json")))
        id_map = {k: v for i, (k, v) in enumerate(id_map.items())}

        g = json.load(open(os.path.join(data_dir, "reddit-G.json")))
        nodes = []
        nodes_type = []
        train_nodes = []
        NODE_TYPE_ID = {"train": 0, "val": 1, "test": 2}
        for nid, node in enumerate(g["nodes"]):
            # id_map[node["id"]] = nid
            nid = id_map[node["id"]]
            if node["test"]:
                ntype = NODE_TYPE_ID["test"]
            elif node["val"]:
                ntype = NODE_TYPE_ID["val"]
            else:
                ntype = NODE_TYPE_ID["train"]
                train_nodes.append(nid)
            nodes.append(nid)
            nodes_type.append(ntype)

        train_neighbors: Dict = {nid: [] for nid in id_map.values()}
        other_neighbors: Dict = {nid: [] for nid in id_map.values()}

        # edges
        np.random.seed(0)
        edge_downsample_mask = (
            np.random.uniform(size=len(g["links"])) < self._edge_downsample_pct
        )
        train_mask = np.zeros(len(id_map), np.bool8)
        train_mask[train_nodes] = True
        for i, e in enumerate(np.array(g["links"])[edge_downsample_mask]):
            src = e["source"]  # id_map[e["source"]]
            tgt = e["target"]  # id_map[e["target"]]
            if train_mask[src] and train_mask[tgt]:
                train_neighbors[src].append(tgt)
                if tgt != src:
                    train_neighbors[tgt].append(src)
            else:
                other_neighbors[src].append(tgt)
                if tgt != src:
                    other_neighbors[tgt].append(src)

        # class map
        fname = os.path.join(data_dir, "reddit-class_map.json")
        class_map = json.load(open(fname))
        class_map = {id_map[k]: onehot(v, 50) for k, v in class_map.items()}

        # feat
        feats = np.load(os.path.join(data_dir, "reddit-feats.npy"))
        train_feats = feats[train_nodes]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)

        return nodes, nodes_type, train_neighbors, other_neighbors, feats, class_map

    def _build_graph(self, output_dir: str) -> str:
        data_dir = output_dir
        raw_data_dir = os.path.join(output_dir, "raw")
        download_file(self.url, raw_data_dir, "reddit.zip")
        fname = os.path.join(raw_data_dir, "reddit.zip")
        with zipfile.ZipFile(fname) as z:
            z.extractall(raw_data_dir)
        d = self._load_raw_graph(os.path.join(raw_data_dir, "reddit"))
        nodes, nodes_type, train_neighbors, other_neighbors, feats, class_map = d
        # assert feats.shape[0] == len(nodes), (feats.shape[0], len(nodes))
        self.NUM_NODES = len(nodes)
        self.FEATURE_DIM = feats.shape[1]
        self.NUM_CLASSES = len(class_map[0])

        graph_file = os.path.join(data_dir, "graph.csv")
        with open(graph_file, "w") as fout:
            for i in range(len(nodes)):
                nid = nodes[i]
                ntype = nodes_type[i]
                nfeat = feats[nid].reshape(-1).tolist()
                label = class_map[nid]
                assert type(nfeat) == list and type(nfeat[0]) == float
                assert type(label) == list
                tmp = self._to_edge_list_node(
                    nid,
                    ntype,
                    nfeat,
                    label,
                    train_neighbor=train_neighbors[nid],
                    train_removed_neighbor=other_neighbors[nid],
                )
                fout.write(tmp)
                fout.write("\n")

        self._write_node_files(data_dir, nodes, nodes_type)
        # convert graph: edge_list -> Binary
        convert.MultiWorkersConverter(
            graph_path=graph_file,
            partition_count=1,
            output_dir=data_dir,
            decoder=decoders.EdgeListDecoder(),
        ).convert()

        return data_dir

    def _write_node_files(self, data_dir, nodes: List[int], nodes_type: List[int]):
        train_file = os.path.join(data_dir, "train.nodes")
        val_file = os.path.join(data_dir, "val.nodes")
        test_file = os.path.join(data_dir, "test.nodes")
        with open(train_file, "w") as fout_train, open(val_file, "w") as fout_val, open(
            test_file, "w"
        ) as fout_test:
            for nid, ntype in zip(nodes, nodes_type):
                if ntype == 0:
                    fout_train.write(str(nid) + "\n")
                elif ntype == 1:
                    fout_val.write(str(nid) + "\n")
                elif ntype == 2:
                    fout_test.write(str(nid) + "\n")

    def _to_edge_list_node(
        self,
        node_id: int,
        node_type: int,
        feat: List[float],
        label: List[float],
        train_neighbor: List[int],
        train_removed_neighbor: List[int],
    ):
        output = ""
        output += f"{node_id},-1,{node_type},1.0,float32,{len(label)},{','.join([str(v) for v in label])},float32,{len(feat)},{','.join([str(v) for v in feat])}\n"
        for nb in sorted(train_neighbor):
            output += f"{node_id},0,{nb},1.0\n"
        for nb in sorted(train_removed_neighbor):
            output += f"{node_id},1,{nb},1.0\n"
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/tmp/reddit", type=str)
    parser.add_argument("--edge_downsample_pct", default=0.1, type=float)

    args = parser.parse_args()

    g = Reddit(args.data_dir, args.edge_downsample_pct)

    print(g.node_features([1], np.array([[0, 50]]), np.float32))
    print(g.node_features([1], np.array([[1, 300]]), np.float32))
