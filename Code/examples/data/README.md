# Training data

Put `graphs.npz` and `labels.npz` in `examples/data/train/`. This is the single data entry used by `examples/train.py`, `examples/resume.py`, validation and export. The NPZ files are local training inputs and are ignored by Git.

Run `examples/prepare_split.py` once per dataset to create `train/split.json` before full training. The manifest records input hashes and fixed train/validation/test design IDs. Identical netlist hashes stay together; parent-circuit grouping is not verified. Existing manifests are validated and preserved. After replacing both NPZs with the same schema, archive the previous manifest with the old data and prepare a new one. Never replace the split during an experiment.
