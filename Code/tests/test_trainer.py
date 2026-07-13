"""End-to-end lifecycle tests for the DARE-Gate trainer."""

import tempfile
import unittest
from pathlib import Path

import torch

from Code.daregate import DAREGateConfig, DAREGateModel
from Code.daregate.training import (
    DAREGateBatch,
    DAREGateTrainer,
    ObjectiveConfig,
    TrainerConfig,
)


def training_batch() -> DAREGateBatch:
    gate_type = torch.tensor([0, 0, 1, 2, 1], dtype=torch.long)
    edge_index = torch.tensor(
        [[0, 1, 0, 2, 3], [2, 2, 3, 4, 4]], dtype=torch.long
    )
    forward_level = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    backward_level = torch.tensor([2, 2, 1, 1, 0], dtype=torch.long)
    rc_source = torch.tensor([-1, -1, -1, -1, 0], dtype=torch.long)
    raw = torch.randn(5, 8, 2) * 0.05
    signature = (raw - raw.mean(dim=1, keepdim=True)).reshape(5, 16)
    return DAREGateBatch(
        gate_type=gate_type,
        edge_index=edge_index,
        forward_level=forward_level,
        backward_level=backward_level,
        rc_source=rc_source,
        signature=signature,
        batch_index=torch.zeros(5, dtype=torch.long),
        names=("tiny_rc",),
    )


def tiny_model() -> DAREGateModel:
    return DAREGateModel(
        DAREGateConfig(
            structure_dim=32,
            function_dim=32,
            behavior_dim=32,
            gate_embedding_dim=8,
            probe_embedding_dim=8,
            attention_heads=4,
            cross_probe_heads=4,
            cross_probe_ff_dim=64,
            mlp_dropout=0.0,
        )
    )


class DAREGateTrainerTest(unittest.TestCase):
    def test_fit_checkpoint_and_resume(self) -> None:
        torch.manual_seed(11)
        batch = training_batch()
        with tempfile.TemporaryDirectory() as directory:
            trainer_config = TrainerConfig(
                learning_rate=1e-3,
                device="cpu",
                use_amp=False,
                checkpoint_dir=directory,
                log_interval=0,
            )
            objective_config = ObjectiveConfig(
                auxiliary_warmup_epochs=0,
                max_similarity_pairs=16,
            )
            trainer = DAREGateTrainer(
                tiny_model(),
                trainer_config=trainer_config,
                objective_config=objective_config,
            )
            history = trainer.fit([batch], [batch], num_epochs=2)
            self.assertEqual(len(history), 2)
            self.assertEqual(trainer.start_epoch, 2)
            self.assertTrue(Path(directory, "model_best.pt").is_file())
            self.assertTrue(Path(directory, "model_last.pt").is_file())
            self.assertTrue(torch.isfinite(torch.tensor(history[-1]["train"]["loss"])))

            resumed = DAREGateTrainer(
                tiny_model(),
                trainer_config=trainer_config,
                objective_config=objective_config,
            )
            checkpoint = resumed.load_checkpoint(Path(directory, "model_last.pt"))
            self.assertEqual(resumed.start_epoch, 2)
            self.assertEqual(len(resumed.history), 2)
            self.assertIn("optimizer", checkpoint)
            for expected, actual in zip(
                trainer.model.parameters(), resumed.model.parameters()
            ):
                torch.testing.assert_close(expected, actual)

    def test_mapping_batch_is_supported(self) -> None:
        batch = training_batch()
        trainer = DAREGateTrainer(
            tiny_model(),
            trainer_config=TrainerConfig(device="cpu", use_amp=False, log_interval=0),
            objective_config=ObjectiveConfig(max_similarity_pairs=4),
        )
        mapping = {
            "gate_type": batch.gate_type,
            "edge_index": batch.edge_index,
            "forward_level": batch.forward_level,
            "backward_level": batch.backward_level,
            "rc_source": batch.rc_source,
            "signature": batch.signature,
            "batch_index": batch.batch_index,
        }
        result = trainer.evaluate([mapping], epoch=0)
        self.assertIn("mae", result)
        self.assertGreaterEqual(result["mae"], 0.0)


if __name__ == "__main__":
    unittest.main()

