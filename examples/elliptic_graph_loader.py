"""Load the public Elliptic Bitcoin transaction graph files and run GPU-friendly feature aggregation.

Expected input directory:
  data/elliptic_bitcoin_dataset/

Expected files:
  elliptic_txs_features.csv
  elliptic_txs_classes.csv
  elliptic_txs_edgelist.csv

The dataset is not committed to this repository. Download it from the public dataset host and place the files under the expected directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch


FEATURES_FILE = "elliptic_txs_features.csv"
CLASSES_FILE = "elliptic_txs_classes.csv"
EDGES_FILE = "elliptic_txs_edgelist.csv"


def load_elliptic(data_dir: Path):
    features_path = data_dir / FEATURES_FILE
    classes_path = data_dir / CLASSES_FILE
    edges_path = data_dir / EDGES_FILE

    for path in [features_path, classes_path, edges_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Download Elliptic dataset files first.")

    features_df = pd.read_csv(features_path, header=None)
    classes_df = pd.read_csv(classes_path)
    edges_df = pd.read_csv(edges_path)

    tx_ids = features_df.iloc[:, 0].astype(str)
    id_to_index = {tx_id: idx for idx, tx_id in enumerate(tx_ids)}

    feature_values = torch.tensor(features_df.iloc[:, 1:].values, dtype=torch.float32)

    src = edges_df.iloc[:, 0].astype(str).map(id_to_index)
    dst = edges_df.iloc[:, 1].astype(str).map(id_to_index)
    valid_edges = src.notna() & dst.notna()
    src_tensor = torch.tensor(src[valid_edges].astype(int).values, dtype=torch.long)
    dst_tensor = torch.tensor(dst[valid_edges].astype(int).values, dtype=torch.long)

    labels = classes_df.copy()
    labels.iloc[:, 0] = labels.iloc[:, 0].astype(str)
    labels = labels[labels.iloc[:, 0].isin(id_to_index)]

    return feature_values, src_tensor, dst_tensor, labels


def aggregate_degrees(num_nodes: int, src: torch.Tensor, dst: torch.Tensor, device: str):
    src = src.to(device)
    dst = dst.to(device)
    ones = torch.ones(src.shape[0], device=device)

    out_degree = torch.zeros(num_nodes, device=device)
    in_degree = torch.zeros(num_nodes, device=device)
    out_degree.scatter_add_(0, src, ones)
    in_degree.scatter_add_(0, dst, ones)

    return torch.stack([out_degree, in_degree], dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load Elliptic public graph dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/elliptic_bitcoin_dataset"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    features, src, dst, labels = load_elliptic(args.data_dir)
    degree_features = aggregate_degrees(features.shape[0], src, dst, device=device)

    print("Device:", device)
    print("Node feature matrix:", tuple(features.shape))
    print("Edges:", int(src.shape[0]))
    print("Degree feature matrix:", tuple(degree_features.shape))
    print("Label rows:", len(labels))


if __name__ == "__main__":
    main()
