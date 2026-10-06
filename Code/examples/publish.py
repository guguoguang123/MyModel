"""Explicitly publish a completed experiment as a reusable model artifact."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triwave_gnn.pretrained import DEFAULT_MODEL, publish_pretrained


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    print(publish_pretrained(args.run, args.output))


if __name__ == "__main__":
    main()
