"""Inspect replacement graph/label bundles without starting training."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triwave_gnn.dataset import CircuitDataset
from triwave_gnn.utils.config import DEFAULT_DATA_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    dataset = CircuitDataset(args.data)
    graph, labels = dataset.graph, dataset.label
    report = {
        "schema_version": str(graph["schema_version"]),
        "design_count": len(dataset),
        "node_count": len(graph["node_role"]),
        "net_edge_count": graph["net_edge_index"].shape[1],
        "cell_edge_count": graph["cell_edge_index"].shape[1],
        "hs_pair_count": labels["hs_pair_node_index"].shape[1],
        "hf_pair_count": labels["hf_tt_pair_node_index"].shape[1],
        "source_sha256": dataset.source_sha256,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
