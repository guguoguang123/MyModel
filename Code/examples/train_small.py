"""Run a bounded three-stage learning check before full pretraining."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.train import main
from triwave_gnn.utils.logger import info


# Defaults for VS Code's Run Python File action. CLI flags can override them.
SMALL_CONFIG = {
    "split_mode": "random", # Keep this learning check independent of the formal split.
    "max_designs": 30,       # Total designs across train, validation and test.
    "max_nodes": 10000,      # Skip larger circuits for this initial check.
    "batch_size": 1,
    "stage1_epochs": 3,
    "stage2_epochs": 3,
    "stage3_epochs": 3,
    "progress_interval": 100,
}


if __name__ == "__main__":
    defaults = []
    for key, value in SMALL_CONFIG.items():
        defaults.extend(("--" + key.replace("_", "-"), str(value)))
    info("Starting small learning check; using SMALL_CONFIG defaults with CLI overrides")
    main([*defaults, *sys.argv[1:]])
