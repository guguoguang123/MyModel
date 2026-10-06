"""Export Hs/Hf/Hb embeddings aligned to graph node IDs for downstream tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triwave_gnn.dataset import CircuitDataset
from triwave_gnn.utils.config import DEFAULT_DATA_DIR
from triwave_gnn.utils.inference import load_model
from triwave_gnn.trainer import move


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--design", action="append", default=None,
                        help="export only this design ID; can be repeated")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    dataset = CircuitDataset(args.data)
    model, state = load_model(args.checkpoint, dataset, args.device)
    output_dir = args.output or args.checkpoint.parent / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    names = args.design or dataset.ids
    unknown = sorted(set(names) - set(dataset.ids))
    if unknown:
        raise ValueError(f"unknown design IDs: {unknown}")
    with torch.no_grad():
        for name in names:
            index = dataset.ids.index(name)
            item = move(dataset[index], args.device)
            hs, hf, hb = model(item)
            node_start, node_stop = map(int, dataset.graph["node_ptr"][index:index + 2])
            path = output_dir / (name + ".npz")
            np.savez_compressed(
                path,
                design_id=np.array(name),
                node_id=np.arange(node_stop - node_start, dtype=np.int64),
                hs=hs.cpu().numpy(),
                hf=hf.cpu().numpy(),
                hb=hb.cpu().numpy(),
                node_mapping_sha256=dataset.graph["node_mapping_sha256"][index],
                netlist_sha256=dataset.graph["netlist_sha256"][index],
                checkpoint_stage=np.array(state["stage"]),
            )
            print(path, flush=True)


if __name__ == "__main__":
    main()
