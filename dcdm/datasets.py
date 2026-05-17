"""Dataset loading, validation, and reporting helpers.

The original script loaded two ``.pt`` files inline.  That works for quick
experiments, but a project-style codebase benefits from a small boundary around
disk layout and graph sanity checks.  These helpers keep the training script
focused on orchestration while still preserving the original dataset format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import statistics


@dataclass(frozen=True)
class GraphDatasetPaths:
    """Expected train/test dataset locations."""

    graph_dir: Path
    train_path: Path
    test_path: Path


@dataclass(frozen=True)
class DatasetSummary:
    """Human-readable facts about a graph dataset."""

    name: str
    size: int
    checked_samples: int
    avg_nodes: float
    avg_edges: float
    min_edges: int
    max_edges: int
    has_user_pos: bool
    has_device_pos: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class DatasetBundle:
    """Loaded datasets plus their resolved paths and summaries."""

    train: Any
    test: Any
    paths: GraphDatasetPaths
    train_summary: DatasetSummary
    test_summary: DatasetSummary


def resolve_dataset_paths(graph_dir: Path) -> GraphDatasetPaths:
    """Resolve the canonical graph dataset files."""

    graph_dir = Path(graph_dir).expanduser().resolve()
    return GraphDatasetPaths(
        graph_dir=graph_dir,
        train_path=graph_dir / "G0_dataset.pt",
        test_path=graph_dir / "G0_dataset_test.pt",
    )


def require_dataset_files(paths: GraphDatasetPaths) -> None:
    """Raise a helpful error when one of the expected files is missing."""

    missing = [path for path in (paths.train_path, paths.test_path) if not path.exists()]
    if missing:
        pretty = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing dataset file(s):\n{pretty}")


def load_graph_datasets(graph_dir: Path, weights_only: bool = False) -> tuple[Any, Any, GraphDatasetPaths]:
    """Load the original train/test PyTorch datasets."""

    import torch

    paths = resolve_dataset_paths(graph_dir)
    require_dataset_files(paths)
    train = torch.load(paths.train_path, weights_only=weights_only)
    test = torch.load(paths.test_path, weights_only=weights_only)
    return train, test, paths


def load_dataset_bundle(graph_dir: Path, weights_only: bool = False, max_check_samples: int = 50) -> DatasetBundle:
    """Load datasets and immediately build lightweight summaries."""

    train, test, paths = load_graph_datasets(graph_dir, weights_only=weights_only)
    train_summary = summarize_dataset(train, "train", max_samples=max_check_samples)
    test_summary = summarize_dataset(test, "test", max_samples=max_check_samples)
    return DatasetBundle(
        train=train,
        test=test,
        paths=paths,
        train_summary=train_summary,
        test_summary=test_summary,
    )


def _safe_len(dataset: Any) -> int:
    try:
        return int(len(dataset))
    except Exception:
        return 0


def _iter_preview(dataset: Any, max_samples: int) -> Iterable[tuple[int, Any]]:
    limit = min(_safe_len(dataset), max_samples)
    for index in range(limit):
        try:
            yield index, dataset[index]
        except Exception as exc:
            yield index, exc


def _sample_num_nodes(sample: Any) -> Optional[int]:
    if hasattr(sample, "num_nodes") and sample.num_nodes is not None:
        return int(sample.num_nodes)
    if hasattr(sample, "x"):
        return int(sample.x.shape[0])
    if hasattr(sample, "device_pos"):
        return int(sample.device_pos.shape[0])
    return None


def _sample_num_edges(sample: Any) -> Optional[int]:
    if hasattr(sample, "edge_index"):
        return int(sample.edge_index.shape[1])
    if hasattr(sample, "edge_attr"):
        return int(sample.edge_attr.shape[0])
    return None


def validate_graph_sample(sample: Any, index: int) -> list[str]:
    """Return warnings for common graph-data shape problems."""

    warnings: list[str] = []
    if isinstance(sample, Exception):
        return [f"sample {index}: failed to read sample: {sample}"]

    for attr in ("x", "edge_index", "edge_attr", "user_pos", "device_pos"):
        if not hasattr(sample, attr):
            warnings.append(f"sample {index}: missing attribute '{attr}'")

    if hasattr(sample, "x") and getattr(sample.x, "dim", lambda: 0)() != 2:
        warnings.append(f"sample {index}: x should be a 2D tensor")

    if hasattr(sample, "edge_index") and sample.edge_index.shape[0] != 2:
        warnings.append(f"sample {index}: edge_index should have shape [2, E]")

    if hasattr(sample, "edge_index") and hasattr(sample, "edge_attr"):
        if sample.edge_index.shape[1] != sample.edge_attr.shape[0]:
            warnings.append(f"sample {index}: edge_index and edge_attr edge counts differ")

    if hasattr(sample, "device_pos") and sample.device_pos.shape[-1] != 2:
        warnings.append(f"sample {index}: device_pos should contain 2D coordinates")

    if hasattr(sample, "user_pos") and sample.user_pos.shape[-1] != 2:
        warnings.append(f"sample {index}: user_pos should contain 2D coordinates")

    return warnings


def summarize_dataset(dataset: Any, name: str, max_samples: int = 50) -> DatasetSummary:
    """Inspect a bounded number of samples and summarize the dataset."""

    size = _safe_len(dataset)
    node_counts: list[int] = []
    edge_counts: list[int] = []
    warnings: list[str] = []
    has_user_pos = False
    has_device_pos = False

    for index, sample in _iter_preview(dataset, max_samples):
        if isinstance(sample, Exception):
            warnings.extend(validate_graph_sample(sample, index))
            continue

        sample_warnings = validate_graph_sample(sample, index)
        warnings.extend(sample_warnings)
        if hasattr(sample, "user_pos"):
            has_user_pos = True
        if hasattr(sample, "device_pos"):
            has_device_pos = True

        num_nodes = _sample_num_nodes(sample)
        num_edges = _sample_num_edges(sample)
        if num_nodes is not None:
            node_counts.append(num_nodes)
        if num_edges is not None:
            edge_counts.append(num_edges)

    checked = min(size, max_samples)
    avg_nodes = statistics.fmean(node_counts) if node_counts else 0.0
    avg_edges = statistics.fmean(edge_counts) if edge_counts else 0.0
    min_edges = min(edge_counts) if edge_counts else 0
    max_edges = max(edge_counts) if edge_counts else 0

    return DatasetSummary(
        name=name,
        size=size,
        checked_samples=checked,
        avg_nodes=avg_nodes,
        avg_edges=avg_edges,
        min_edges=min_edges,
        max_edges=max_edges,
        has_user_pos=has_user_pos,
        has_device_pos=has_device_pos,
        warnings=tuple(warnings[:20]),
    )


def format_dataset_summary(summary: DatasetSummary) -> str:
    """Turn a dataset summary into a concise report block."""

    lines = [
        f"{summary.name} dataset:",
        f"  size: {summary.size}",
        f"  checked samples: {summary.checked_samples}",
        f"  avg nodes: {summary.avg_nodes:.2f}",
        f"  avg edges: {summary.avg_edges:.2f}",
        f"  edge range: {summary.min_edges}..{summary.max_edges}",
        f"  has user positions: {summary.has_user_pos}",
        f"  has device positions: {summary.has_device_pos}",
    ]
    if summary.warnings:
        lines.append("  warnings:")
        lines.extend(f"    - {warning}" for warning in summary.warnings)
    return "\n".join(lines)


def print_dataset_report(bundle: DatasetBundle) -> None:
    """Print train/test dataset summaries with their source directory."""

    print("\n=== Dataset Report ===")
    print(f"Graph directory: {bundle.paths.graph_dir}")
    print(format_dataset_summary(bundle.train_summary))
    print(format_dataset_summary(bundle.test_summary))


def dataset_sizes(bundle: DatasetBundle) -> dict[str, int]:
    """Return train/test sizes for logging and checkpoint metadata."""

    return {
        "train_size": bundle.train_summary.size,
        "test_size": bundle.test_summary.size,
    }


def pick_preview_indices(dataset: Sequence[Any], count: int) -> list[int]:
    """Return evenly spaced sample indices for deterministic visual checks."""

    size = len(dataset)
    if count <= 0 or size <= 0:
        return []
    if count >= size:
        return list(range(size))
    if count == 1:
        return [0]
    step = (size - 1) / (count - 1)
    return [round(i * step) for i in range(count)]

