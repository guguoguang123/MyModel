"""Report interpretable masked errors for a saved checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triwave_gnn.dataset import CircuitDataset
from triwave_gnn.utils.config import DEFAULT_DATA_DIR
from triwave_gnn.utils.inference import load_model
from triwave_gnn.utils.losses import BEHAVIOR, functional_distance
from triwave_gnn.utils.utils import get_function_acc
from triwave_gnn.trainer import move
from triwave_gnn.utils.model_utils import model_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    dataset = CircuitDataset(args.data)
    model, checkpoint = load_model(args.run / "best.pt", dataset, args.device)
    if checkpoint.get("source_sha256") is not None and checkpoint["source_sha256"] != dataset.source_sha256:
        raise ValueError("evaluation NPZ files differ from those used in this run")
    splits = json.loads((args.run / "split.json").read_text())
    results = {}
    with torch.no_grad():
        for split_name, names in splits.items():
            errors = {}
            counts = {}
            def collect(key, prediction, target, mask):
                mask = mask & torch.isfinite(target)
                if bool(mask.any()):
                    errors[key] = errors.get(key, 0.0) + torch.abs(prediction[mask] - target[mask]).sum().item()
                    counts[key] = counts.get(key, 0) + int(mask.sum())
            for name in names:
                item = move(dataset[dataset.ids.index(name)], args.device)
                output = model_outputs(model, item)
                rank_acc, rank_count = get_function_acc(output["hf"], item["hf_pairs"],
                                                        item["hf_tt_target"], item["hf_tt_mask"])
                if rank_count:
                    errors["hf_tt_rank_accuracy"] = errors.get("hf_tt_rank_accuracy", 0.0) + rank_acc * rank_count
                    counts["hf_tt_rank_accuracy"] = counts.get("hf_tt_rank_accuracy", 0) + rank_count
                collect("hf_prob_mae", output["hf_prob_logits"].sigmoid(), item["hf_prob"], torch.ones_like(item["hf_prob"], dtype=torch.bool))
                collect("hf_tt_cosine_distance_mae", functional_distance(output["hf"], item["hf_pairs"]), item["hf_tt_target"], item["hf_tt_mask"])
                if item["hs_mask"].any():
                    pred = (output["hs_rc_logits"].sigmoid() >= 0.5).float()
                    collect("hs_rc_error_rate", pred, item["hs_target"], item["hs_mask"])
                for j, (key, mask_key, kind) in enumerate(BEHAVIOR):
                    pred = output["hb_raw"][:, j].sigmoid() if kind == "probability" else torch.expm1(output["hb_raw"][:, j]).clamp_min(0)
                    collect(key + "_mae", pred, item[key], item[mask_key])
            results[split_name] = {
                key: {("accuracy" if key == "hf_tt_rank_accuracy" else "mae"):
                      errors[key] / counts[key], "count": counts[key]}
                for key in errors
            }
    (args.run / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["test"], indent=2))


if __name__ == "__main__":
    main()
