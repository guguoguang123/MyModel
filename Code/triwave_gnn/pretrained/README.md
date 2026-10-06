# Published weights

`model.pth` is created with `python examples/publish.py --run examples/exp/<run>` after a completed stage-3 training run has been selected for reuse. A training run keeps `last.pt` for resuming and `best.pt` for final selection. The current one-epoch run checks the workflow and is not automatically published as a pretrained model.
