"""Load one circuit and export its Hs, Hf and Hb embeddings."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triwave_gnn import NpzParser, TriWaveGNN
from triwave_gnn.trainer import move
from triwave_gnn.utils.checkpoint import load
from triwave_gnn.utils.inference import load_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "examples" / "data" / "train"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--design", required=True, help="design ID in graphs.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset = NpzParser(args.data).dataset
    if args.design not in dataset.ids:
        raise ValueError("unknown design ID: " + args.design)
    saved = load(args.checkpoint, "cpu")
    model = TriWaveGNN(len(dataset.logic_vocab), len(dataset.role_vocab),
                       len(dataset.pin_vocab), hidden=saved["config"]["hidden"])
    model, _ = load_model(args.checkpoint, dataset, args.device, model=model)
    design_index = dataset.ids.index(args.design)
    item = move(dataset[design_index], args.device)
    with torch.no_grad():
        hs, hf, hb = model(item)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, design_id=np.array(args.design),
                        node_id=np.arange(item["pin"].num_nodes, dtype=np.int64),
                        node_mapping_sha256=dataset.graph["node_mapping_sha256"][design_index],
                        netlist_sha256=dataset.graph["netlist_sha256"][design_index],
                        hs=hs.cpu().numpy(),
                        hf=hf.cpu().numpy(),
                        hb=hb.cpu().numpy())
    print(args.output)


if __name__ == "__main__":
    main()
