"""Published TriWave-GNN model artifact management."""

from pathlib import Path

from ..utils.checkpoint import atomic_save, load


DEFAULT_MODEL = Path(__file__).with_name("model.pth")


def publish_pretrained(run_dir, output=DEFAULT_MODEL):
    """Publish a completed stage-3 run for feature extraction."""
    run_dir = Path(run_dir)
    state = load(run_dir / "best.pt", "cpu")
    last = load(run_dir / "last.pt", "cpu")
    if state.get("stage") != 3 or last.get("stage") != 3 or not last.get("stage_complete", False):
        raise ValueError("only a completed stage-3 run can be published")
    atomic_save(state, output)
    return Path(output)
