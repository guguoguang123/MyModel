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
        probability=torch.tensor([0.5, 0.5, 0.25, 0.5, 0.125]),
        tt_pair_index=torch.tensor(
            [[0, 0, 2], [1, 3, 4]], dtype=torch.long
        ),
        tt_distance=torch.tensor([0.8, 0.4, 0.2]),
        min_tt_distance=torch.tensor([0.2, 0.1, 0.05]),
        rc_pair_index=torch.tensor(
            [[0, 0, 1, 2], [2, 3, 4, 4]], dtype=torch.long
        ),
        is_reconvergent=torch.tensor([1.0, 1.0, 0.0, 0.0]),
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
                output_dir=str(Path(directory, "output")),
                log_interval=0,
                save_every=1,
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
            self.assertTrue(Path(directory, "model_last.pt").is_file())
            self.assertTrue(torch.isfinite(torch.tensor(history[-1]["train"]["loss"])))
            output_root = Path(directory, "output")
            self.assertTrue(Path(output_root, "logs", "history.json").is_file())
            self.assertTrue(Path(output_root, "metrics", "epoch_0002.json").is_file())
            saved = trainer.export_outputs([batch], split="validation")
            self.assertEqual(len(saved), 1)
            self.assertTrue(saved[0][0].is_file())
            self.assertTrue(saved[0][1].is_file())
            self.assertIn("stage", history[-1])
            self.assertEqual(history[-1]["stage"], "joint")

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

    def test_train_loader_switches_at_configured_epoch(self) -> None:
        trainer = DAREGateTrainer(
            tiny_model(),
            trainer_config=TrainerConfig(device="cpu", use_amp=False),
            objective_config=ObjectiveConfig(max_similarity_pairs=0),
        )
        early_loader = object()
        late_loader = object()
        observed: list[tuple[object, int]] = []

        def fake_train_epoch(loader: object, epoch: int) -> dict[str, float]:
            observed.append((loader, epoch))
            return {"loss": 1.0, "mae": 1.0}

        trainer.train_epoch = fake_train_epoch  # type: ignore[method-assign]
        trainer.fit(
            early_loader,
            None,
            num_epochs=4,
            late_train_loader=late_loader,
            train_loader_switch_epoch=2,
        )
        self.assertEqual(
            observed,
            [(early_loader, 0), (early_loader, 1),
             (late_loader, 2), (late_loader, 3)],
        )

    def test_checkpoint_is_not_written_off_period(self) -> None:
        batch = training_batch()
        with tempfile.TemporaryDirectory() as directory:
            trainer = DAREGateTrainer(
                tiny_model(),
                trainer_config=TrainerConfig(
                    device="cpu",
                    use_amp=False,
                    checkpoint_dir=directory,
                    save_every=5,
                    log_interval=0,
                ),
                objective_config=ObjectiveConfig(max_similarity_pairs=4),
            )
            trainer.fit([batch], None, num_epochs=1)
            self.assertFalse(Path(directory, "model_last.pt").exists())
            self.assertEqual(list(Path(directory).glob("model_best*.pt")), [])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_amp_training_step(self) -> None:
        """Exercise model indexing, BCE and GradScaler under CUDA autocast."""
        torch.manual_seed(13)
        with tempfile.TemporaryDirectory() as directory:
            trainer = DAREGateTrainer(
                tiny_model(),
                trainer_config=TrainerConfig(
                    device="cuda",
                    use_amp=True,
                    output_dir=directory,
                    log_interval=0,
                ),
                objective_config=ObjectiveConfig(
                    functionality_start_epoch=0,
                    behavior_start_epoch=0,
                    max_similarity_pairs=4,
                ),
            )
            result = trainer.train_epoch([training_batch()], epoch=0)
        self.assertTrue(torch.isfinite(torch.tensor(result["loss"])))
        self.assertGreater(result["loss"], 0.0)


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
            "probability": batch.probability,
            "tt_pair_index": batch.tt_pair_index,
            "tt_distance": batch.tt_distance,
            "rc_pair_index": batch.rc_pair_index,
            "is_reconvergent": batch.is_reconvergent,
            "min_tt_distance": batch.min_tt_distance,
        }
        result = trainer.evaluate([mapping], epoch=0)
        self.assertIn("mae", result)
        self.assertGreaterEqual(result["mae"], 0.0)


if __name__ == "__main__":
    unittest.main()

