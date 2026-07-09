"""PyTorch Geometric GNN baseline for the public Elliptic Bitcoin transaction graph dataset.

Expected input directory:
  data/elliptic_bitcoin_dataset/

Expected files:
  elliptic_txs_features.csv
  elliptic_txs_classes.csv
  elliptic_txs_edgelist.csv

This example requires torch_geometric. It is not included in requirements.txt because installation depends on the installed PyTorch and CUDA versions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn


FEATURES_FILE = "elliptic_txs_features.csv"
CLASSES_FILE = "elliptic_txs_classes.csv"
EDGES_FILE = "elliptic_txs_edgelist.csv"


class GraphSageBaseline(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        try:
            from torch_geometric.nn import SAGEConv
        except ImportError as exc:
            raise SystemExit(
                "This example requires torch_geometric. Install PyTorch Geometric for your "
                "PyTorch/CUDA environment before running."
            ) from exc

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


def load_elliptic_for_pyg(data_dir: Path):
    features_path = data_dir / FEATURES_FILE
    classes_path = data_dir / CLASSES_FILE
    edges_path = data_dir / EDGES_FILE

    for path in [features_path, classes_path, edges_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")

    features_df = pd.read_csv(features_path, header=None)
    classes_df = pd.read_csv(classes_path)
    edges_df = pd.read_csv(edges_path)

    tx_ids = features_df.iloc[:, 0].astype(str)
    id_to_index = {tx_id: idx for idx, tx_id in enumerate(tx_ids)}

    x = torch.tensor(features_df.iloc[:, 1:].values, dtype=torch.float32)
    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)

    src = edges_df.iloc[:, 0].astype(str).map(id_to_index)
    dst = edges_df.iloc[:, 1].astype(str).map(id_to_index)
    valid = src.notna() & dst.notna()
    edge_index = torch.tensor(
        [src[valid].astype(int).values, dst[valid].astype(int).values], dtype=torch.long
    )

    # Elliptic labels are commonly represented as 1=illicit, 2=licit, unknown=unknown.
    # We train only on known licit/illicit labels and map licit->0, illicit->1.
    label_tx_col = classes_df.columns[0]
    label_col = classes_df.columns[1]
    y = torch.full((x.shape[0],), -1, dtype=torch.long)

    for _, row in classes_df.iterrows():
        tx_id = str(row[label_tx_col])
        label = str(row[label_col])
        if tx_id not in id_to_index:
            continue
        if label == "1":
            y[id_to_index[tx_id]] = 1
        elif label == "2":
            y[id_to_index[tx_id]] = 0

    train_mask = y >= 0
    return x, edge_index, y, train_mask


def train(data_dir: Path, epochs: int, hidden_channels: int) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, edge_index, y, train_mask = load_elliptic_for_pyg(data_dir)

    x = x.to(device)
    edge_index = edge_index.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)

    model = GraphSageBaseline(x.shape[1], hidden_channels, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = loss_fn(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            with torch.no_grad():
                pred = logits[train_mask].argmax(dim=1)
                acc = (pred == y[train_mask]).float().mean().item()
            print(f"epoch={epoch:03d} loss={loss.item():.4f} known-label-accuracy={acc:.4f}")

    print("Device:", device)
    print("Nodes:", x.shape[0])
    print("Edges:", edge_index.shape[1])
    print("Known labels:", int(train_mask.sum().item()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PyG GraphSAGE baseline on Elliptic dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/elliptic_bitcoin_dataset"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-channels", type=int, default=64)
    args = parser.parse_args()
    train(args.data_dir, args.epochs, args.hidden_channels)


if __name__ == "__main__":
    main()
