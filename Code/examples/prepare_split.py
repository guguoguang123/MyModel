"""Freeze the current NPZ dataset split without starting model training."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triwave_gnn.utils.config import DEFAULT_DATA_DIR
from triwave_gnn.utils.split_utils import bundle_identity, make_fixed_split, validate_fixed_split


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare a fixed whole-design training split")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    path = args.output or args.data / "split.json"
    print(f"checking bundle metadata and hashes: {args.data}", flush=True)
    ids, hashes, sources = bundle_identity(args.data)
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        validate_fixed_split(manifest, ids, hashes, sources)
        print(f"keeping existing fixed split: {path}", flush=True)
    else:
        manifest = make_fixed_split(ids, hashes, sources, args.seed)
        validate_fixed_split(manifest, ids, hashes, sources)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents accidentally replacing an established split.
        with path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    counts = {name: len(values) for name, values in manifest["splits"].items()}
    print(f"split={path} counts={counts}", flush=True)
    return manifest


if __name__ == "__main__":
    main()
