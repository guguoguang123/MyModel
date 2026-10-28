"""Append-only training timeline with elapsed time and ETA."""

import json
import os
import re
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


def _color_enabled():
    return (sys.stdout.isatty() and os.environ.get("TERM") != "dumb"
            and "NO_COLOR" not in os.environ)


def _paint(line, color):
    if not color:
        return line
    line = line.replace("[INFO]", "\033[1;33m[INFO]\033[0m", 1)
    return re.sub(r"\[([━●]+)(─*)\]", lambda match:
                  "[\033[96m" + match[1] + "\033[90m" + match[2] + "\033[0m]", line)


def info(message):
    """Prefix permanent training messages; redirected logs contain no ANSI codes."""
    for line in str(message).splitlines():
        print(_paint("[INFO] " + line, _color_enabled()), flush=True)


def _clip(line, width):
    return line if len(line) <= width else line[:max(0, width - 3)] + "." * min(3, width)


METRIC_LABELS = {
    "prob_mae": ("ProbMAE", "P"),
    "rc_accuracy": ("RCAcc", "RC"),
    "tt_distance_mae": ("TTMAE", "TT"),
    "tt_rank_accuracy": ("TTRank", "Rank"),
    "hb_tp_mae": ("TPMAE", "TP"),
    "hb_first_mean_mae": ("FirstMAE", "First"),
    "hb_stable_mean_mae": ("StableMAE", "Stable"),
    "hb_glitch_rate_mae": ("GlitchMAE", "Glitch"),
}


class BatchProgress:
    """One status message on two rows, with a continuous playback-style track."""

    def __init__(self, phase, stage, epoch, total, detail_interval, epoch_total=None,
                 show_loss=True):
        if total <= 0:
            raise ValueError("progress needs at least one batch")
        self.phase, self.stage, self.epoch = phase, stage, epoch
        self.total = total
        self.detail_interval = detail_interval
        self.epoch_total = epoch_total
        self.show_loss = show_loss
        self.terminal = sys.stdout.isatty() and os.environ.get("TERM") != "dumb"
        self.color = _color_enabled()
        self.drawn = False
        self.started = time.monotonic()
        self.last_draw = None

    def __enter__(self):
        if self.terminal:
            self.update(0, None, {}, {})
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.finish()

    def _track(self, current, width):
        position = min(width - 1, max(0, int((width - 1) * current / self.total)))
        return "[" + "━" * position + "●" + "─" * (width - position - 1) + "]"

    def _lines(self, current, loss, terms, metrics, elapsed, width):
        eta = format_seconds(elapsed / current * (self.total - current)) if current else "--:--:--"
        epoch = f" E{self.epoch}" if self.epoch else ""
        if epoch and self.epoch_total:
            epoch += f"/{self.epoch_total}"
        stage = f" S{self.stage}/3" if self.stage is not None else ""
        header = f"[INFO] {self.phase.upper()}{stage}{epoch} "
        suffix = (f" {current}/{self.total} {current / self.total:5.1%} "
                  f"{format_seconds(elapsed)} ETA {eta}")
        # Shorten labels/times before reducing the bar, preserving two rows.
        if len(header) + len(suffix) + 10 > width:
            stage = f" S{self.stage}" if self.stage is not None else ""
            header = f"[INFO] {self.phase.upper()}{stage}" + (f" E{self.epoch} " if epoch else " ")
            suffix = (f" {current}/{self.total} {current / self.total:.0%} "
                      f"{format_seconds(elapsed)[3:] if elapsed < 3600 else format_seconds(elapsed)} "
                      f"ETA {eta[3:] if eta.startswith('00:') else eta}")
        bar_width = max(1, min(24, width - len(header) - len(suffix) - 2))
        progress_line = _clip(header + self._track(current, bar_width) + suffix, width)

        def tokens(compact=False):
            result = [f"Loss={loss:.3f}" if compact else f"Loss={loss:.4f}"] if loss is not None else ["Loss=--"]
            if not self.show_loss:
                result = []
            for key, value in metrics.items():
                labels = METRIC_LABELS.get(key, (key, key))
                label = labels[int(compact)]
                number = f"{value:.1%}" if "accuracy" in key else f"{value:.3f}" if compact else f"{value:.4f}"
                result.append(f"{label}={number}")
            return result

        values, separator = tokens(), " | "
        indent = " " * len("[INFO] ")
        if len(indent + separator.join(values)) > width:
            values, separator = tokens(compact=True), " "
        metric_line = indent + separator.join(values)
        if len(metric_line) > width:
            kept = []
            for value in values:
                if len(indent + separator.join([*kept, value]) + " ...") > width:
                    break
                kept.append(value)
            metric_line = indent + separator.join(kept) + " ..."
        # On wider terminals also show the individual optimization losses.
        extra = " | " + " ".join(f"{key}Loss={value:.4f}" for key, value in terms.items())
        if terms and len(metric_line + extra) <= width:
            metric_line += extra
        return progress_line, _clip(metric_line, width)

    def update(self, current, loss, terms, metrics):
        now = time.monotonic()
        if self.terminal:
            # Redraw at most five times per second; no permanent batch messages.
            if current != self.total and self.last_draw is not None and now - self.last_draw < 0.2:
                return
        elif current != self.total and not (self.detail_interval and current and current % self.detail_interval == 0):
            return
        width = max(1, shutil.get_terminal_size((120, 24)).columns - 1)
        lines = self._lines(current, loss, terms, metrics, now - self.started, width)
        if self.terminal:
            if self.drawn:
                sys.stdout.write("\r\033[1A")
            sys.stdout.write("\r\033[2K" + _paint(lines[0], self.color)
                             + "\n\r\033[2K" + _paint(lines[1], self.color))
            sys.stdout.flush()
            self.drawn = True
        else:
            print("\n".join(lines), flush=True)
        self.last_draw = now

    def finish(self):
        if self.drawn:
            # Remove the temporary panel; epoch summaries remain in scrollback.
            sys.stdout.write("\r\033[2K\033[1A\r\033[2K")
            sys.stdout.flush()
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
