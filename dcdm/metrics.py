"""Metrics, history, and terminal formatting utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import csv
import math
import time


@dataclass(frozen=True)
class RewardSummary:
    """Aggregate reward statistics for one sampling pass."""

    count: int
    valid_count: int
    total: float
    mean: float
    valid_mean: float
    minimum: float
    maximum: float
    zero_count: int

    @property
    def valid_ratio(self) -> float:
        if self.count == 0:
            return 0.0
        return self.valid_count / self.count


@dataclass(frozen=True)
class EpochRecord:
    """One row of training history."""

    epoch: int
    loss: float
    reward_mean: float
    reward_valid_mean: float
    reward_valid_count: int
    reward_count: int
    reward_min: float
    reward_max: float
    reward_valid_ratio: float
    elapsed_sec: float
    learning_rate: float


class Stopwatch:
    """Simple elapsed-time helper for training loops."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def reset(self) -> None:
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._start


def summarize_rewards(rewards: Iterable[float]) -> RewardSummary:
    """Compute summary statistics while treating zero rewards as invalid."""

    values = [float(value) for value in rewards]
    valid = [value for value in values if value != 0]
    count = len(values)
    valid_count = len(valid)
    total = sum(values)
    mean = total / count if count else 0.0
    valid_mean = sum(valid) / valid_count if valid_count else 0.0
    minimum = min(values) if values else 0.0
    maximum = max(values) if values else 0.0
    zero_count = count - valid_count
    return RewardSummary(
        count=count,
        valid_count=valid_count,
        total=total,
        mean=mean,
        valid_mean=valid_mean,
        minimum=minimum,
        maximum=maximum,
        zero_count=zero_count,
    )


def rolling_mean(values: Iterable[float], window: int) -> list[float]:
    """Compute a trailing rolling mean without external dependencies."""

    values = [float(value) for value in values]
    if window <= 1:
        return values
    output: list[float] = []
    running = 0.0
    queue: list[float] = []
    for value in values:
        queue.append(value)
        running += value
        if len(queue) > window:
            running -= queue.pop(0)
        output.append(running / len(queue))
    return output


def current_learning_rate(optimizer) -> float:
    """Read the first optimizer learning rate."""

    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def is_better_reward(candidate: float, best: Optional[float]) -> bool:
    """Return True when a reward should replace the current best score."""

    if best is None:
        return True
    if math.isnan(best):
        return True
    return candidate > best


def make_epoch_record(
    epoch: int,
    loss: float,
    rewards: Iterable[float],
    elapsed_sec: float,
    learning_rate: float,
) -> EpochRecord:
    """Build a history row from raw epoch outputs."""

    summary = summarize_rewards(rewards)
    return EpochRecord(
        epoch=epoch,
        loss=float(loss),
        reward_mean=summary.mean,
        reward_valid_mean=summary.valid_mean,
        reward_valid_count=summary.valid_count,
        reward_count=summary.count,
        reward_min=summary.minimum,
        reward_max=summary.maximum,
        reward_valid_ratio=summary.valid_ratio,
        elapsed_sec=float(elapsed_sec),
        learning_rate=float(learning_rate),
    )


def format_epoch_record(record: EpochRecord) -> str:
    """Create the single-line epoch message printed during training."""

    if record.reward_valid_count > 0:
        reward = f"{record.reward_valid_mean:.4f}"
    else:
        reward = "invalid"
    return (
        f"Epoch {record.epoch:3d} | "
        f"Loss: {record.loss:.4f} | "
        f"Valid Avg Reward: {reward} | "
        f"Valid: {record.reward_valid_count}/{record.reward_count} | "
        f"LR: {record.learning_rate:.2e} | "
        f"Elapsed: {record.elapsed_sec:.1f}s"
    )


class TrainingHistory:
    """In-memory epoch history with CSV persistence."""

    fieldnames = [
        "epoch",
        "loss",
        "reward_mean",
        "reward_valid_mean",
        "reward_valid_count",
        "reward_count",
        "reward_min",
        "reward_max",
        "reward_valid_ratio",
        "elapsed_sec",
        "learning_rate",
    ]

    def __init__(self) -> None:
        self.records: list[EpochRecord] = []

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def append(self, record: EpochRecord) -> None:
        self.records.append(record)

    def latest(self) -> Optional[EpochRecord]:
        if not self.records:
            return None
        return self.records[-1]

    def best(self) -> Optional[EpochRecord]:
        if not self.records:
            return None
        return max(self.records, key=lambda record: record.reward_valid_mean)

    def total_elapsed(self) -> float:
        return sum(record.elapsed_sec for record in self.records)

    def reward_curve(self, valid_only: bool = True) -> list[float]:
        if valid_only:
            return [record.reward_valid_mean for record in self.records]
        return [record.reward_mean for record in self.records]

    def losses(self) -> list[float]:
        return [record.loss for record in self.records]

    def to_rows(self) -> list[dict[str, object]]:
        return [asdict(record) for record in self.records]

    def save_csv(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            for row in self.to_rows():
                writer.writerow(row)

    @classmethod
    def load_csv(cls, path: Path) -> "TrainingHistory":
        history = cls()
        path = Path(path)
        if not path.exists():
            return history
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                history.append(
                    EpochRecord(
                        epoch=int(raw["epoch"]),
                        loss=float(raw["loss"]),
                        reward_mean=float(raw["reward_mean"]),
                        reward_valid_mean=float(raw["reward_valid_mean"]),
                        reward_valid_count=int(raw["reward_valid_count"]),
                        reward_count=int(raw["reward_count"]),
                        reward_min=float(raw["reward_min"]),
                        reward_max=float(raw["reward_max"]),
                        reward_valid_ratio=float(raw["reward_valid_ratio"]),
                        elapsed_sec=float(raw["elapsed_sec"]),
                        learning_rate=float(raw["learning_rate"]),
                    )
                )
        return history

    def short_report(self) -> str:
        if not self.records:
            return "No training history recorded."
        latest = self.latest()
        best = self.best()
        assert latest is not None
        assert best is not None
        return "\n".join(
            [
                "Training history:",
                f"  epochs: {len(self.records)}",
                f"  latest reward: {latest.reward_valid_mean:.4f}",
                f"  best reward: {best.reward_valid_mean:.4f} at epoch {best.epoch}",
                f"  total elapsed: {self.total_elapsed():.1f}s",
            ]
        )


def write_history_csv(records: Iterable[EpochRecord], path: Path) -> None:
    """Write arbitrary records to CSV."""

    history = TrainingHistory()
    for record in records:
        history.append(record)
    history.save_csv(path)

