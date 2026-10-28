"""Fixed membership and input identity checks without running model training."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from triwave_gnn import NpzParser
from triwave_gnn.tests.fixtures import make_bundle
from triwave_gnn.utils.config import training_parser
from triwave_gnn.utils.split_utils import (
    bundle_identity, make_fixed_split, resolve_split_file, validate_fixed_split,
)


class FixedSplit(unittest.TestCase):
    def setUp(self):
        self.ids = [f"design_{i}" for i in range(20)]
        self.hashes = [f"hash_{i // 2}" for i in range(20)]
        self.sources = {"graphs.npz": "graph-hash", "labels.npz": "label-hash"}
        self.manifest = make_fixed_split(self.ids, self.hashes, self.sources)

    def test_deterministic_complete_split_groups_duplicate_netlists(self):
        self.assertEqual(self.manifest, make_fixed_split(self.ids, self.hashes, self.sources))
        splits = validate_fixed_split(self.manifest, self.ids, self.hashes, self.sources)
        self.assertEqual(sorted(i for values in splits.values() for i in values), list(range(20)))
        owner = {i: key for key, values in splits.items() for i in values}
        for i in range(0, 20, 2):
            self.assertEqual(owner[i], owner[i + 1])

    def test_rejects_changed_inputs_or_membership(self):
        with self.assertRaisesRegex(ValueError, "NPZ hashes differ"):
            validate_fixed_split(self.manifest, self.ids, self.hashes, {})
        for replacement in ([], ["unknown"], self.manifest["splits"]["train"]):
            changed = copy.deepcopy(self.manifest)
            changed["splits"]["test"] = replacement
            with self.assertRaises(ValueError):
                validate_fixed_split(changed, self.ids, self.hashes, self.sources)

    def test_rejects_duplicate_netlists_across_partitions(self):
        changed = copy.deepcopy(self.manifest)
        # Swap single IDs across partitions while preserving complete coverage.
        train, test = changed["splits"]["train"], changed["splits"]["test"]
        train[0], test[0] = test[0], train[0]
        with self.assertRaisesRegex(ValueError, "same netlist hash"):
            validate_fixed_split(changed, self.ids, self.hashes, self.sources)

    def test_default_entry_requires_manifest_and_random_mode_allows_sampling(self):
        with tempfile.TemporaryDirectory() as scratch:
            args = training_parser().parse_args(["--data", scratch])
            with self.assertRaises(FileNotFoundError):
                resolve_split_file(args)
            path = Path(scratch) / "split.json"
            path.write_text("{}", encoding="utf-8")
            self.assertEqual(resolve_split_file(args), path.resolve())
            args.max_designs = 30
            with self.assertRaisesRegex(ValueError, "sampling limits"):
                resolve_split_file(args)
            args.split_mode = "random"
            self.assertIsNone(resolve_split_file(args))
            args.split_file = path
            with self.assertRaisesRegex(ValueError, "requires"):
                resolve_split_file(args)

    def test_npz_parser_uses_saved_membership_and_prepare_preserves_manifest(self):
        from examples.prepare_split import main as prepare

        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            make_bundle(directory)
            manifest = prepare(["--data", scratch])
            path = directory / "split.json"
            original = path.read_bytes()
            prepare(["--data", scratch, "--seed", "99"])
            self.assertEqual(path.read_bytes(), original)
            ids, hashes, sources = bundle_identity(directory)
            parser = NpzParser(directory, split_file=path)
            self.assertEqual(parser.get_split_indices(),
                             validate_fixed_split(manifest, ids, hashes, sources))
            changed = json.loads(original)
            changed["source_sha256"]["labels.npz"] = "changed"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "NPZ hashes differ"):
                prepare(["--data", scratch])


if __name__ == "__main__":
    unittest.main()
