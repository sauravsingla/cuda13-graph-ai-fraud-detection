"""RAPIDS cuGraph example for the public Elliptic Bitcoin transaction graph dataset.

Expected input directory:
  data/elliptic_bitcoin_dataset/

Expected files:
  elliptic_txs_features.csv
  elliptic_txs_classes.csv
  elliptic_txs_edgelist.csv

This example requires a RAPIDS environment with cudf and cugraph installed.
It is intentionally separate from requirements.txt because RAPIDS installation depends on CUDA, Python, and platform versions.
"""

from __future__ import annotations

import argparse
from pathlib import Path


FEATURES_FILE = "elliptic_txs_features.csv"
CLASSES_FILE = "elliptic_txs_classes.csv"
EDGES_FILE = "elliptic_txs_edgelist.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cuGraph features on Elliptic dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/elliptic_bitcoin_dataset"))
    args = parser.parse_args()

    try:
        import cudf
        import cugraph
    except ImportError as exc:
        raise SystemExit(
            "This example requires RAPIDS cudf and cugraph. "
            "Install RAPIDS for your CUDA/Python environment before running."
        ) from exc

    edges_path = args.data_dir / EDGES_FILE
    features_path = args.data_dir / FEATURES_FILE
    classes_path = args.data_dir / CLASSES_FILE

    for path in [edges_path, features_path, classes_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")

    edges = cudf.read_csv(edges_path)
    features = cudf.read_csv(features_path, header=None)
    classes = cudf.read_csv(classes_path)

    src_col = edges.columns[0]
    dst_col = edges.columns[1]

    graph = cugraph.Graph(directed=True)
    graph.from_cudf_edgelist(edges, source=src_col, destination=dst_col, renumber=True)

    pagerank = cugraph.pagerank(graph)
    indegree = graph.in_degree()
    outdegree = graph.out_degree()

    print("Elliptic cuGraph summary")
    print("Nodes in feature file:", len(features))
    print("Edges:", len(edges))
    print("Class rows:", len(classes))
    print("PageRank rows:", len(pagerank))
    print("In-degree rows:", len(indegree))
    print("Out-degree rows:", len(outdegree))
    print("Sample PageRank:")
    print(pagerank.head())


if __name__ == "__main__":
    main()
