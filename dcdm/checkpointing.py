"""Checkpoint helpers for long D-CDM training runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .metrics import TrainingHistory, is_better_reward


@dataclass(frozen=True)
class CheckpointMetadata:
    """Metadata stored next to model and optimizer states."""

    epoch: int
    best_reward: Optional[float]
    reward_baseline: float
    config: dict[str, Any]


def config_to_dict(config: Any) -> dict[str, Any]:
    """Serialize public scalar/list config values."""

    values: dict[str, Any] = {}
    for name in sorted(dir(config)):
        if name.startswith("_"):
            continue
        value = getattr(config, name)
        if callable(value):
            continue
        if isinstance(value, (str, int, float, bool, list, tuple)):
            values[name] = value
    return values


def checkpoint_filename(epoch: int) -> str:
    """Return a stable checkpoint filename for one epoch."""

    return f"checkpoint_epoch_{epoch:04d}.pt"


def save_checkpoint(
    trainer: Any,
    epoch: int,
    history: TrainingHistory,
    checkpoints_dir: Path,
    best_reward: Optional[float],
    config: Any,
    tag: Optional[str] = None,
) -> Path:
    """Save model, optimizer, baseline, and history state."""

    import torch

    checkpoints_dir = Path(checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{tag}.pt" if tag else checkpoint_filename(epoch)
    path = checkpoints_dir / filename
    payload = {
        "metadata": {
            "epoch": int(epoch),
            "best_reward": best_reward,
            "reward_baseline": float(getattr(trainer, "reward_baseline", 0.0)),
            "config": config_to_dict(config),
        },
        "model_state_dict": trainer.denoising_net.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "history": history.to_rows(),
    }
    torch.save(payload, path)
    return path


def load_checkpoint(trainer: Any, path: Path, map_location: Optional[str] = None, strict: bool = True) -> dict[str, Any]:
    """Load a checkpoint into an existing trainer instance."""

    import torch

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    trainer.denoising_net.load_state_dict(payload["model_state_dict"], strict=strict)
    if "optimizer_state_dict" in payload:
        trainer.optimizer.load_state_dict(payload["optimizer_state_dict"])
    metadata = payload.get("metadata", {})
    if "reward_baseline" in metadata:
        trainer.reward_baseline = float(metadata["reward_baseline"])
    return payload


def history_from_checkpoint_payload(payload: dict[str, Any]) -> TrainingHistory:
    """Reconstruct a TrainingHistory object from checkpoint rows."""

    from .metrics import EpochRecord

    history = TrainingHistory()
    for raw in payload.get("history", []):
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


def checkpoint_epoch(path: Path) -> int:
    """Best-effort epoch parsing from a checkpoint filename."""

    stem = Path(path).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else -1


def list_checkpoints(checkpoints_dir: Path) -> list[Path]:
    """List epoch checkpoints in ascending order."""

    checkpoints_dir = Path(checkpoints_dir)
    if not checkpoints_dir.exists():
        return []
    paths = sorted(checkpoints_dir.glob("checkpoint_epoch_*.pt"), key=checkpoint_epoch)
    return paths


def find_latest_checkpoint(checkpoints_dir: Path) -> Optional[Path]:
    """Return the newest epoch checkpoint, if one exists."""

    checkpoints = list_checkpoints(checkpoints_dir)
    if not checkpoints:
        return None
    return checkpoints[-1]


def prune_checkpoints(checkpoints_dir: Path, keep: int) -> list[Path]:
    """Delete older epoch checkpoints and return the removed paths."""

    if keep <= 0:
        return []
    checkpoints = list_checkpoints(checkpoints_dir)
    stale = checkpoints[:-keep]
    removed: list[Path] = []
    for path in stale:
        try:
            path.unlink()
            removed.append(path)
        except FileNotFoundError:
            pass
    return removed


def load_training_state(
    trainer: Any,
    checkpoint_path: Path,
    map_location: Optional[str] = None,
) -> tuple[int, Optional[float], TrainingHistory]:
    """Load checkpoint state and return next epoch, best reward, history."""

    payload = load_checkpoint(trainer, checkpoint_path, map_location=map_location)
    metadata = payload.get("metadata", {})
    epoch = int(metadata.get("epoch", 0))
    best_reward = metadata.get("best_reward")
    if best_reward is not None:
        best_reward = float(best_reward)
    history = history_from_checkpoint_payload(payload)
    return epoch + 1, best_reward, history


def save_if_best(
    trainer: Any,
    epoch: int,
    history: TrainingHistory,
    checkpoints_dir: Path,
    best_reward: Optional[float],
    candidate_reward: float,
    config: Any,
) -> tuple[Optional[float], Optional[Path]]:
    """Save a best-model checkpoint when the candidate reward improves."""

    if not is_better_reward(candidate_reward, best_reward):
        return best_reward, None
    path = save_checkpoint(
        trainer=trainer,
        epoch=epoch,
        history=history,
        checkpoints_dir=checkpoints_dir,
        best_reward=candidate_reward,
        config=config,
        tag="best",
    )
    return candidate_reward, path


def save_final_model(trainer: Any, model_path: Path) -> Path:
    """Save only the denoising network weights for downstream use."""

    import torch

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.denoising_net.state_dict(), model_path)
    return model_path

