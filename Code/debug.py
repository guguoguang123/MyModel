"""Inspect a trained TriWave-GNN on one supplied pin-level circuit."""

import argparse
from pathlib import Path

import torch

from triwave_gnn import CircuitDataset, TriWaveGNN
from triwave_gnn.trainer import move
from triwave_gnn.utils.model_utils import model_outputs
from triwave_gnn.utils.checkpoint import load
from triwave_gnn.utils.config import DEFAULT_DATA_DIR
from triwave_gnn.utils.inference import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    dataset = CircuitDataset(args.data)
    if args.design not in dataset.ids:
        raise ValueError("unknown design ID: " + args.design)
    saved = load(args.checkpoint, "cpu")
    model = TriWaveGNN(len(dataset.logic_vocab), len(dataset.role_vocab),
                       len(dataset.pin_vocab), saved["config"]["hidden"])
    model, _ = load_model(args.checkpoint, dataset, args.device, model=model)
    item = move(dataset[dataset.ids.index(args.design)], args.device)
    with torch.no_grad():
        output = model_outputs(model, item)
    print("design:", args.design)
    print("embeddings:", {name: tuple(output[name].shape) for name in ("hs", "hf", "hb")})
    print("probability range:", output["hf_prob_logits"].sigmoid().min().item(),
          output["hf_prob_logits"].sigmoid().max().item())
    print("reconvergence pairs:", item["hs_pairs"].shape[1])


if __name__ == "__main__":
    main()
