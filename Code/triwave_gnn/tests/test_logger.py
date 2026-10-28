"""Check bounded log output and the two-row terminal panel without training."""

import io
import os
import unittest
from unittest.mock import patch

from triwave_gnn.utils.logger import BatchProgress, METRIC_LABELS, info


class TerminalBuffer(io.StringIO):
    def isatty(self):
        return True


class TrainingDisplay(unittest.TestCase):
    def test_evaluation_panel_reuses_style_without_training_stage_or_loss(self):
        progress = BatchProgress("test", None, 0, 1663, 100, show_loss=False)
        for width in (60, 80, 120):
            lines = progress._lines(100, None, {}, {"prob_mae": .02}, 50, width)
            self.assertEqual(len(lines), 2)
            self.assertTrue(all(len(line) <= width for line in lines))
            self.assertIn("[INFO] TEST", lines[0])
            self.assertIn("ETA", lines[0])
            self.assertNotIn("SNone", lines[0])
            self.assertNotIn("Loss=", lines[1])
            self.assertIn("ProbMAE=", lines[1])

    def test_metrics_and_framed_progress_stay_within_two_rows(self):
        progress = BatchProgress("train", 3, 5, 243, 100, epoch_total=40)
        metrics = {key: 0.85 if "accuracy" in key else 0.123 for key in METRIC_LABELS}
        for width in (24, 60, 80, 120, 180):
            lines = progress._lines(120, 0.25, {"prob": 0.1}, metrics, 61, width)
            self.assertEqual(len(lines), 2)
            self.assertTrue(all(len(line) <= width for line in lines))
            self.assertTrue(lines[0].startswith("[INFO]"))
            self.assertTrue(lines[1].startswith("       "))
            self.assertEqual("\n".join(lines).count("[INFO]"), 1)
            if width >= 60:
                self.assertIn("━", lines[0])
                self.assertIn("─", lines[0])
                self.assertIn("●", lines[0])
                self.assertIn("Loss=", lines[1])
        self.assertIn("Glitch=", progress._lines(120, 0.25, {}, metrics, 61, 120)[1])

    def test_terminal_refresh_reuses_rows_and_adds_yellow_info_prefix(self):
        stream = TerminalBuffer()
        with patch("sys.stdout", stream), patch.dict(os.environ, {"TERM": "xterm"}, clear=True), \
                patch("triwave_gnn.utils.logger.time.monotonic", return_value=0.0) as clock:
            with BatchProgress("train", 1, 1, 3, 1, epoch_total=30) as progress:
                for current in range(1, 4):
                    clock.return_value = current * 0.3
                    progress.update(current, 0.5 / current, {}, {"rc_accuracy": 0.75})
            info("Epoch complete")
        output = stream.getvalue()
        self.assertIn("\033[1;33m[INFO]\033[0m", output)
        self.assertIn("E1/30", output)
        self.assertEqual(output.count("\033[1A"), 4)  # Three redraws and panel cleanup.
        self.assertIn("RCAcc=75.0%", output)
        self.assertNotIn("losses:", output)
        self.assertFalse(progress.drawn)

    def test_fast_batches_are_throttled_but_completion_always_draws(self):
        stream = TerminalBuffer()
        with patch("sys.stdout", stream), patch.dict(os.environ, {"TERM": "xterm"}, clear=True), \
                patch("triwave_gnn.utils.logger.time.monotonic", return_value=0.0):
            with BatchProgress("val", 1, 1, 50, 1) as progress:
                for current in range(1, 51):
                    progress.update(current, 0.1, {}, {})
        self.assertEqual(stream.getvalue().count("Loss="), 2)  # Initial and final frames.
        self.assertIn("50/50", stream.getvalue())

    def test_redirected_output_has_few_plain_snapshots(self):
        stream = io.StringIO()
        with patch("sys.stdout", stream):
            with BatchProgress("train", 1, 1, 243, 100) as progress:
                for current in range(1, 244):
                    progress.update(current, 0.1, {}, {"prob_mae": 0.2})
        output = stream.getvalue()
        self.assertEqual(len(output.splitlines()), 6)
        self.assertEqual(output.count("[INFO]"), 3)
        self.assertNotIn("\033", output)
        for batch in (100, 200, 243):
            self.assertIn(f"{batch}/243", output)

    def test_exception_clears_panel_and_propagates(self):
        stream = TerminalBuffer()
        with patch("sys.stdout", stream), patch.dict(os.environ, {"TERM": "xterm"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                with BatchProgress("train", 1, 1, 243, 100) as progress:
                    raise RuntimeError("interrupted")
        self.assertTrue(stream.getvalue().endswith("\r\033[2K\033[1A\r\033[2K"))
        self.assertFalse(progress.drawn)


if __name__ == "__main__":
    unittest.main()
