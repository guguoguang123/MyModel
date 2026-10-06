"""Append-only training timeline with elapsed time and ETA."""

import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone


class AverageMeter:
    """Track the running average of a batch metric."""

    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, weight=1):
        self.total += float(value) * weight
        self.count += weight

    @property
    def avg(self):
        return self.total / max(self.count, 1)


def format_seconds(seconds):
    minutes, second = divmod(max(0, int(seconds)), 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours:02d}:{minute:02d}:{second:02d}"


class BatchProgress:
    """Thin progress track that redraws in a terminal and snapshots to logs."""

    def __init__(self, phase, stage, epoch, total, detail_interval):
        self.phase, self.stage, self.epoch = phase, stage, epoch
        self.total = total
        self.detail_interval = detail_interval
        self.terminal = sys.stdout.isatty() and os.environ.get("TERM") != "dumb"
        self.color = self.terminal
        self.drawn = False
        self.started = time.monotonic()

    def _track(self, current, width):
        filled = min(width, max(0, round(width * current / self.total)))
        active = "─" * filled
        remaining = "─" * (width - filled)
        if self.color:
            return f"\033[96m{active}\033[90m{remaining}\033[0m"
        return active + "·" * (width - filled)

    def update(self, current, loss, terms, metrics):
        detail = self.detail_interval and (
            current % self.detail_interval == 0 or current == self.total
        )
        if not (self.terminal or detail):
            return
        elapsed = time.monotonic() - self.started
        eta = elapsed / current * (self.total - current)
        width = max(30, min(shutil.get_terminal_size((100, 24)).columns - 2, 120))
        line = (f"{self.phase.upper()}  Stage {self.stage}/3  Epoch {self.epoch}  "
                f"{current}/{self.total}  {current / self.total:.0%}  "
                f"Loss {loss:.4f}  {format_seconds(elapsed)} / ETA {format_seconds(eta)}")
        if self.terminal:
            if self.drawn:
                sys.stdout.write("\r\033[1A")
            sys.stdout.write("\r\033[2K" + line[:width] + "\n\033[2K" + self._track(current, width))
            sys.stdout.flush()
            self.drawn = True
            if detail:
                print(flush=True)
                self.drawn = False
        else:
            print(line, flush=True)
            print(self._track(current, min(width, 80)), flush=True)
        if detail:
            print("  losses: " + " ".join(f"{key}={value:.4f}" for key, value in terms.items()), flush=True)
            print("  metrics: " + " ".join(f"{key}={value:.4f}" for key, value in metrics.items()), flush=True)

    def finish(self):
        if self.drawn:
            print(flush=True)
            self.drawn = False


class TrainingLogger:
    def __init__(self, run_dir, completed=0, total=1):
        self.path = run_dir / "timeline.jsonl"
        self.started = time.monotonic()
        self.completed = completed
        self.total = max(total, 1)

    def epoch(self, stage, epoch, train, val):
        self.completed += 1
        elapsed = time.monotonic() - self.started
        rate = elapsed / max(self.completed, 1)
        eta = max(self.total - self.completed, 0) * rate
        record = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage, "epoch": epoch,
            "train_loss": train["mean_loss"], "val_loss": val["mean_loss"],
            "train_terms": train["terms"], "val_terms": val["terms"],
            "train_metrics": train["metrics"], "val_metrics": val["metrics"],
            "elapsed_seconds": round(elapsed, 3),
            "eta_seconds": round(eta, 3),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        return record
