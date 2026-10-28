"""Check threshold selection and AUC against exhaustive pairwise definitions."""

import contextlib
import io
import unittest
import warnings

import torch

from triwave_gnn.utils.functional_equivalence import (
    equivalence_samples, select_threshold, equivalence_metrics,
)


class FunctionalEquivalence(unittest.TestCase):
    def setUp(self):
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)
        self.warning_context = warnings.catch_warnings()
        self.warning_context.__enter__()
        warnings.simplefilter("ignore", UserWarning)
        self.addCleanup(self.warning_context.__exit__, None, None, None)

    def test_mask_finite_and_zero_tolerance(self):
        target = torch.tensor([0., 1e-9, 1e-7, float("nan"), float("inf"), 0.])
        distance = torch.tensor([0.1, 0.2, 0.3, float("nan"), 0., 0.], requires_grad=True)
        scores, labels = equivalence_samples(distance, target, torch.tensor([1, 1, 1, 1, 1, 0]).bool())
        self.assertEqual(labels.tolist(), [True, True, False])
        self.assertEqual(scores.device.type, "cpu")
        self.assertFalse(scores.requires_grad)
        torch.testing.assert_close(scores, torch.tensor([.9, .8, .7]))

    def test_tied_scores_are_indivisible(self):
        scores = torch.tensor([.9, .9, .8, .1])
        labels = torch.tensor([True, False, True, False])
        threshold = select_threshold(scores, labels)
        self.assertEqual(threshold, float(scores[2]))
        result = equivalence_metrics(scores, labels, threshold)
        self.assertEqual([result[k] for k in ('tp', 'fp', 'tn', 'fn')], [2, 1, 1, 0])
        self.assertAlmostEqual(result['precision'], 2 / 3)
        self.assertAlmostEqual(result['recall'], 1.)
        self.assertAlmostEqual(result['f1'], .8)
        self.assertAlmostEqual(result['roc_auc'], .625)

    def test_exhaustive_threshold_and_pairwise_auc_oracles(self):
        generator = torch.Generator().manual_seed(73)
        for _ in range(30):
            scores = torch.randint(0, 5, (20,), generator=generator).double() / 4
            labels = torch.randint(0, 2, (20,), generator=generator).bool()
            labels[0], labels[1] = True, False
            candidates = []
            for threshold in scores.unique().tolist():
                predicted = scores >= threshold
                tp = int((predicted & labels).sum())
                f1 = 2 * tp / (int(predicted.sum()) + int(labels.sum()))
                candidates.append((f1, threshold))
            expected = max(candidates)[1]
            selected = select_threshold(scores, labels)
            self.assertEqual(selected, expected)
            result = equivalence_metrics(scores, labels, selected)
            differences = scores[labels, None] - scores[~labels][None, :]
            expected_auc = ((differences > 0).double() + .5 * (differences == 0)).mean()
            self.assertAlmostEqual(result['roc_auc'], float(expected_auc))

    def test_all_equal_and_no_predicted_positives(self):
        scores = torch.ones(4)
        labels = torch.tensor([True, False, False, False])
        self.assertEqual(select_threshold(scores, labels), 1.)
        self.assertEqual(equivalence_metrics(scores, labels, 1.)['roc_auc'], .5)
        result = equivalence_metrics(scores, labels, 2.)
        self.assertEqual(result['f1'], 0.)
        self.assertEqual(result['precision'], 0.)

    def test_missing_pairs_and_classes(self):
        with self.assertRaisesRegex(ValueError, 'no valid functional pairs'):
            select_threshold(torch.empty(0), torch.empty(0, dtype=torch.bool))
        for labels in (torch.zeros(3, dtype=torch.bool), torch.ones(3, dtype=torch.bool)):
            scores = torch.tensor([.1, .2, .3])
            self.assertIsNone(select_threshold(scores, labels))
            result = equivalence_metrics(scores, labels, .2)
            self.assertIsNone(result['f1'])
            self.assertIsNone(result['roc_auc'])
            self.assertEqual(sum(result[key] for key in ('tp', 'fp', 'tn', 'fn')), 3)
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            equivalence_samples(torch.tensor([float('nan')]), torch.zeros(1), torch.ones(1, dtype=torch.bool))

    def test_test_data_does_not_change_validation_threshold(self):
        val_scores = torch.tensor([.9, .8, .7, .1])
        val_labels = torch.tensor([True, True, False, False])
        threshold = select_threshold(val_scores, val_labels)
        test_scores = torch.tensor([.6, .5, .4, .3])
        test_labels = torch.tensor([True, True, False, False])
        result = equivalence_metrics(test_scores, test_labels, threshold)
        self.assertEqual(result['threshold'], threshold)
        self.assertEqual(result['f1'], 0.)
        self.assertNotEqual(select_threshold(test_scores, test_labels), threshold)


class EvaluationEntrypoint(unittest.TestCase):
    def test_checkpoint_selection_aliases_and_legacy_metrics(self):
        import importlib.util
        import json
        import sys
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from triwave_gnn.dataset import CircuitDataset
        from triwave_gnn.model import TriWaveGNN
        from triwave_gnn.tests.fixtures import make_bundle

        entry = Path(__file__).resolve().parents[2] / "examples" / "evaluate.py"
        spec = importlib.util.spec_from_file_location("evaluation_entry", entry)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_bundle(root)
            dataset = CircuitDataset(root)
            model = TriWaveGNN(len(dataset.logic_vocab), len(dataset.role_vocab),
                               len(dataset.pin_vocab), hidden=4)
            splits = dict(train=[dataset.ids[0]], val=[dataset.ids[1]], test=[dataset.ids[2]])
            state = dict(architecture_id=model.architecture_id, model=model.state_dict(),
                         config={"hidden": 4}, logic_vocab=dataset.logic_vocab,
                         role_vocab=dataset.role_vocab, pin_vocab=dataset.pin_vocab,
                         feature_normalization={key: {"mean": getattr(dataset, key + "_mean"),
                                                       "std": getattr(dataset, key + "_std")}
                                                for key in ("net", "cell")},
                         source_sha256=dataset.source_sha256, split_ids=splits, stage=3, epoch=1)
            torch.save(state, root / "best.pt")
            (root / "split.json").write_text(json.dumps(splits))
            original_text = json.dumps({"train": {"sentinel": "original preserved"}})
            (root / "metrics.json").write_text(original_text)
            for mode in ("best", "fallback", "explicit"):
                with self.subTest(mode=mode):
                    if mode == "fallback":
                        (root / "best.pt").rename(root / "last.pt")
                        aliased = dict(splits)
                        aliased["validation"] = aliased.pop("val")
                        (root / "split.json").write_text(json.dumps(aliased))
                    argv = [str(entry), "--data", str(root), "--run", str(root), "--device", "cpu"]
                    if mode == "explicit":
                        (root / "last.pt").rename(root / "chosen.pt")
                        argv.extend(["--checkpoint", str(root / "chosen.pt")])
                    terminal = io.StringIO()
                    with patch.object(sys, "argv", argv), contextlib.redirect_stdout(terminal), warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        module.main()
                    result = json.loads((root / "metrics_extended.json").read_text())
                    self.assertEqual((root / "metrics.json").read_text(), original_text)
                    self.assertEqual(result["previous_metrics"]["train"]["sentinel"], "original preserved")
                    expected = {"hf_prob_mae", "hs_rc_error_rate", "hf_tt_cosine_distance_mae",
                                "hf_tt_rank_accuracy", "hb_tp_mae", "hb_first_mean_mae",
                                "hb_stable_mean_mae", "hb_glitch_rate_mae"}
                    self.assertTrue(expected <= result["test"].keys())
                    self.assertIn("hs_rc_accuracy", result["test"])
                    self.assertEqual(result["test"]["hs_rc_accuracy"]["accuracy"],
                                     1 - result["test"]["hs_rc_error_rate"]["mae"])
                    for key in expected | {"hs_rc_accuracy"}:
                        self.assertIn("count", result["test"][key])
                    self.assertNotIn("Loss=", terminal.getvalue())
                    self.assertIn("[INFO] TEST", terminal.getvalue())
                    self.assertNotIn("inference_time", result["test"])
                    row = result["test"]["per_circuit"][0]
                    self.assertTrue(expected <= row.keys())
                    self.assertEqual(row["functional_equivalence"]["positive_count"], 1)
                    self.assertEqual(result["test"]["functional_equivalence_macro"]["contributing_circuits"]["roc_auc"], 1)
                    self.assertNotIn("similarity_diagnostics", result["test"])
                    self.assertNotIn("SIMILARITY DIAGNOSTICS", terminal.getvalue())
                    self.assertNotIn("DeepGate2", terminal.getvalue())
                    self.assertNotIn("Audit difference", terminal.getvalue())
                    for label in ("precision=", "recall=", "f1=", "accuracy=", "roc_auc=", "TP=", "FP=", "TN=", "FN=", "positive_count=", "negative_count=", "threshold="):
                        self.assertIn(label, terminal.getvalue())
                    self.assertIn("%", terminal.getvalue())
                    self.assertEqual(result["evaluation_metadata"]["evaluated_splits"],
                                     ["val" if mode == "best" else "validation", "test"])
                    self.assertFalse((root / "equivalence_test_scores.npz").exists())
                    val_name = "val" if mode == "best" else "validation"
                    functional = result["test"]["functional_equivalence"]
                    self.assertEqual(functional["threshold"], result[val_name]["functional_equivalence"]["threshold"])
                    self.assertEqual(functional["positive_count"], 1)
                    self.assertEqual(functional["negative_count"], 2)
                    self.assertIsNotNone(functional["f1"])
                    chosen = {"best": "best.pt", "fallback": "last.pt", "explicit": "chosen.pt"}[mode]
                    self.assertEqual(result["evaluation_metadata"]["checkpoint"], str(root / chosen))


if __name__ == '__main__':
    unittest.main()
