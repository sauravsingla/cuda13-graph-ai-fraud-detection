"""PyTorch Geometric GraphSAGE baseline for the Elliptic transaction graph.

Expected input directory:
  data/elliptic_bitcoin_dataset/

This example requires torch_geometric. It is kept outside the base dependency set
because installation depends on the installed PyTorch and CUDA versions.
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


def split_labeled_nodes(
    y: torch.Tensor,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic stratified masks over known labels."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")

    generator = torch.Generator().manual_seed(seed)
    train_mask = torch.zeros_like(y, dtype=torch.bool)
    eval_mask = torch.zeros_like(y, dtype=torch.bool)

    for class_value in (0, 1):
        indices = torch.nonzero(y == class_value, as_tuple=False).squeeze(1)
        if indices.numel() < 2:
            raise ValueError(f"Class {class_value} requires at least two labeled nodes")

        shuffled = indices[torch.randperm(indices.numel(), generator=generator)]
        split_at = max(1, min(int(train_ratio * indices.numel()), indices.numel() - 1))
        train_mask[shuffled[:split_at]] = True
        eval_mask[shuffled[split_at:]] = True

    return train_mask, eval_mask


def load_elliptic_for_pyg(
    data_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    x = (x - x.mean(dim=0)) / x.std(dim=0, unbiased=False).clamp_min(1e-6)

    src = edges_df.iloc[:, 0].astype(str).map(id_to_index)
    dst = edges_df.iloc[:, 1].astype(str).map(id_to_index)
    valid = src.notna() & dst.notna()
    edge_index = torch.tensor(
        [src[valid].astype(int).values, dst[valid].astype(int).values],
        dtype=torch.long,
    )

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

    return x, edge_index, y


def train(data_dir: Path, epochs: int, hidden_channels: int, seed: int) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    x, edge_index, y = load_elliptic_for_pyg(data_dir)
    train_mask, eval_mask = split_labeled_nodes(y, seed=seed)

    x = x.to(device)
    edge_index = edge_index.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    eval_mask = eval_mask.to(device)

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
            model.eval()
            with torch.no_grad():
                eval_logits = model(x, edge_index)
                predictions = eval_logits[eval_mask].argmax(dim=1)
                accuracy = (predictions == y[eval_mask]).float().mean().item()
            print(f"epoch={epoch:03d} loss={loss.item():.4f} holdout-accuracy={accuracy:.4f}")

    print("Device:", device)
    print("Nodes:", x.shape[0])
    print("Edges:", edge_index.shape[1])
    print("Training labels:", int(train_mask.sum().item()))
    print("Holdout labels:", int(eval_mask.sum().item()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GraphSAGE baseline on the Elliptic dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/elliptic_bitcoin_dataset"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.data_dir, args.epochs, args.hidden_channels, args.seed)


if __name__ == "__main__":
    main()
