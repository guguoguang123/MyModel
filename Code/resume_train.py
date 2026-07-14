"""Manual checkpoint-resume entry point for DARE-Gate."""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

if __package__:
    from .daregate.cli.resume import main
else:
    from daregate.cli.resume import main

if __name__ == "__main__":
    raise SystemExit(main())
