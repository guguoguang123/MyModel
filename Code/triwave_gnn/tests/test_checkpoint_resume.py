"""Exercise checkpoint scheduling and interrupted resume with simulated epochs."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from triwave_gnn import NpzParser, TriWaveGNN
from triwave_gnn.tests.fixtures import make_bundle
from triwave_gnn.trainer import TriWaveTrainer
from triwave_gnn.utils.checkpoint import load
from triwave_gnn.utils.config import training_parser


class CheckpointResume(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="checkpoint_test_")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        make_bundle(self.root)
        self.parser = NpzParser(self.root)
        self.train, self.val = self.parser.get_dataset()
        self.args = training_parser().parse_args([
            "--data", str(self.root), "--output", str(self.root / "run"),
            "--split-mode", "random", "--device", "cpu", "--hidden", "4",
            "--stage1-epochs", "12", "--stage2-epochs", "2", "--stage3-epochs", "2",
            "--patience", "0", "--lr-step", "0",
        ])
        self.silence = contextlib.redirect_stdout(io.StringIO())
        self.silence.__enter__()
        self.addCleanup(self.silence.__exit__, None, None, None)

    def trainer(self):
        dataset = self.parser.dataset
        model = TriWaveGNN(len(dataset.logic_vocab), len(dataset.role_vocab),
                           len(dataset.pin_vocab), hidden=4)
        return TriWaveTrainer(model, self.args, dataset, self.parser.get_split_indices())

    @staticmethod
    def simulated_epoch(trainer, interrupt=None, plateau=False):
        def run(indices, stage, training=False, epoch=0, phase=None):
            if training:
                if epoch == interrupt:
                    raise InterruptedError("simulated process interruption")
                with torch.no_grad():
                    trainer.model.readout_prob.bias.fill_(epoch)
            score = 20.0 - (min(epoch, 1) if plateau else epoch)
            return {"mean_loss": score, "terms": {}, "metrics": {}, "per_batch": []}
        return run

    def test_saves_every_five_epochs_and_at_nonmultiple_stage_end(self):
        trainer = self.trainer()
        trainer._run_epoch = self.simulated_epoch(trainer)
        trainer.set_training_args(1)
        with patch.object(trainer, "_save_resume", wraps=trainer._save_resume) as save:
            trainer.train(12, self.train, self.val)
        self.assertEqual([call.args[1] for call in save.call_args_list], [5, 10, 12])
        self.assertTrue(save.call_args_list[-1].kwargs["completed"])
        saved = load(self.root / "run" / "last.pt")
        self.assertEqual(saved["epoch"], 12)
        self.assertEqual(saved["stage_best_checkpoint"]["epoch"], 12)
        self.assertTrue(saved["stage_complete"])

    def test_resume_with_smaller_batch_restores_saved_state_and_replays_unsaved_epochs(self):
        trainer = self.trainer()
        trainer._run_epoch = self.simulated_epoch(trainer, interrupt=8)
        trainer.set_training_args(1)
        with self.assertRaises(InterruptedError):
            trainer.train(12, self.train, self.val)
        path = self.root / "run"
        self.assertEqual(load(path / "last.pt")["epoch"], 5)
        self.assertEqual(load(path / "stage1_best.pt")["epoch"], 7)
        self.args.resume = path / "last.pt"
        self.args.batch_size = 16
        resumed = self.trainer()
        self.assertEqual(resumed.args.batch_size, 16)
        self.assertEqual(resumed.next_epoch, 6)
        self.assertEqual(len(resumed.history), 5)
        self.assertEqual(load(path / "stage1_best.pt")["epoch"], 5)
        self.assertEqual(float(resumed.model.readout_prob.bias.detach()), 5.0)
        resumed._run_epoch = self.simulated_epoch(resumed)
        resumed.set_training_args(1)
        resumed.train(12, self.train, self.val)
        self.assertEqual([row["epoch"] for row in resumed.history], list(range(1, 13)))
        self.assertEqual(load(path / "stage1_best.pt")["epoch"], 12)
        self.assertEqual(load(path / "last.pt")["config"]["batch_size"], 16)

    def test_early_stop_saves_before_first_periodic_checkpoint(self):
        self.args.patience = 2
        trainer = self.trainer()
        trainer._run_epoch = self.simulated_epoch(trainer, plateau=True)
        trainer.set_training_args(1)
        trainer.train(12, self.train, self.val)
        saved = load(self.root / "run" / "last.pt")
        self.assertEqual(saved["epoch"], 3)
        self.assertTrue(saved["stage_complete"])
        self.assertEqual(saved["stage_best_checkpoint"]["epoch"], 1)
        self.assertEqual(float(trainer.model.readout_prob.bias.detach()), 1.0)
        self.args.resume = self.root / "run" / "last.pt"
        self.assertEqual(self.trainer().next_stage, 2)

    def test_invalid_save_interval_is_rejected(self):
        self.args.save_interval = 0
        with self.assertRaisesRegex(ValueError, "save_interval"):
            self.trainer()


if __name__ == "__main__":
    unittest.main()
