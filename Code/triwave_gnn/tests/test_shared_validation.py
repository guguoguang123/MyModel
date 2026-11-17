"""Check that parallel processes create one complete shared baseline."""

import json
import multiprocessing
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ..utils.shared_validation import baseline_lock


def create_or_reuse(directory):
    path = Path(directory) / "baseline.json"
    with baseline_lock(Path(directory) / ".baseline.lock"):
        if not path.exists():
            with (Path(directory) / "creators.txt").open("a") as stream:
                stream.write("created\n")
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps({"complete": True, "scores": list(range(10000))}))
            temporary.replace(path)
        assert json.loads(path.read_text())["complete"]


class SharedBaselineTest(unittest.TestCase):
    def test_three_processes_create_one_baseline(self):
        with TemporaryDirectory() as directory:
            context = multiprocessing.get_context("fork")
            processes = [context.Process(target=create_or_reuse, args=(directory,)) for _ in range(3)]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)
            self.assertEqual((Path(directory) / "creators.txt").read_text(), "created\n")


if __name__ == "__main__":
    unittest.main()
