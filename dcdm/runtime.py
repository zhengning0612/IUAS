"""Runtime helpers for repeatable D-CDM experiments.

This module is intentionally small and dependency-light at import time.  The
training entry point imports it before optional heavy packages are loaded, so
functions that need PyTorch import it locally.  Keeping those imports local
makes commands such as ``python main.py --help`` usable even on machines where
the full research stack has not been installed yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import os
import platform
import random
import sys


@dataclass(frozen=True)
class RuntimeInfo:
    """A compact snapshot of the Python process and host machine."""

    python: str
    executable: str
    platform: str
    cwd: Path
    torch: str = "not imported"
    cuda_available: Optional[bool] = None
    cuda_device_count: Optional[int] = None
    device_name: Optional[str] = None


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved directories used by one experiment run."""

    project_root: Path
    graph_dir: Path
    output_dir: Path
    checkpoints_dir: Path
    figures_dir: Path
    logs_dir: Path

    @classmethod
    def build(cls, project_root: Path, graph_dir: Path, output_dir: Path) -> "ProjectPaths":
        project_root = project_root.resolve()
        graph_dir = graph_dir.resolve()
        output_dir = output_dir.resolve()
        return cls(
            project_root=project_root,
            graph_dir=graph_dir,
            output_dir=output_dir,
            checkpoints_dir=output_dir / "checkpoints",
            figures_dir=output_dir,
            logs_dir=output_dir / "logs",
        )

    @property
    def model_path(self) -> Path:
        return self.output_dir / "dcdm_denoising_net.pth"

    @property
    def history_path(self) -> Path:
        return self.logs_dir / "training_history.csv"

    def figure_path(self, filename: str) -> Path:
        return self.figures_dir / filename


def ensure_directories(paths: ProjectPaths) -> None:
    """Create output directories used by training and reporting."""

    for directory in (paths.output_dir, paths.checkpoints_dir, paths.figures_dir, paths.logs_dir):
        directory.mkdir(parents=True, exist_ok=True)


def set_random_seed(seed: Optional[int], deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch when a seed is provided."""

    if seed is None:
        return

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def get_device(explicit_device: Optional[str] = None, prefer_cuda: bool = True):
    """Return a PyTorch device using a predictable selection policy."""

    import torch

    if explicit_device:
        return torch.device(explicit_device)
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collect_runtime_info() -> RuntimeInfo:
    """Collect runtime details without requiring PyTorch to be installed."""

    info = RuntimeInfo(
        python=sys.version.split()[0],
        executable=sys.executable,
        platform=platform.platform(),
        cwd=Path.cwd(),
    )

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0
        device_name = torch.cuda.get_device_name(0) if cuda_available and device_count else None
        info = RuntimeInfo(
            python=info.python,
            executable=info.executable,
            platform=info.platform,
            cwd=info.cwd,
            torch=torch.__version__,
            cuda_available=cuda_available,
            cuda_device_count=device_count,
            device_name=device_name,
        )
    except Exception:
        pass

    return info


def count_parameters(model: Any, trainable_only: bool = False) -> int:
    """Count model parameters while keeping the call site tidy."""

    parameters = model.parameters()
    if trainable_only:
        parameters = (param for param in parameters if param.requires_grad)
    return int(sum(param.numel() for param in parameters))


def summarize_model(model: Any) -> Mapping[str, int]:
    """Return total and trainable parameter counts."""

    return {
        "parameters_total": count_parameters(model, trainable_only=False),
        "parameters_trainable": count_parameters(model, trainable_only=True),
    }


def format_number(value: float | int) -> str:
    """Format numeric values for short terminal reports."""

    if isinstance(value, int):
        return f"{value:,}"
    if abs(value) >= 1000:
        return f"{value:,.2f}"
    return f"{value:.4f}"


def iter_config_items(config: Any) -> Iterable[tuple[str, Any]]:
    """Yield public scalar-like config attributes in a stable order."""

    for name in sorted(dir(config)):
        if name.startswith("_"):
            continue
        value = getattr(config, name)
        if callable(value):
            continue
        if isinstance(value, (str, int, float, bool, list, tuple)):
            yield name, value


def format_config(config: Any) -> str:
    """Create a readable multi-line configuration block."""

    lines = ["Configuration:"]
    for name, value in iter_config_items(config):
        lines.append(f"  {name}: {value}")
    return "\n".join(lines)


def format_runtime_info(info: RuntimeInfo) -> str:
    """Create a readable multi-line runtime block."""

    lines = [
        "Runtime:",
        f"  Python: {info.python}",
        f"  Executable: {info.executable}",
        f"  Platform: {info.platform}",
        f"  Working directory: {info.cwd}",
        f"  PyTorch: {info.torch}",
    ]
    if info.cuda_available is not None:
        lines.append(f"  CUDA available: {info.cuda_available}")
    if info.cuda_device_count is not None:
        lines.append(f"  CUDA devices: {info.cuda_device_count}")
    if info.device_name:
        lines.append(f"  Device name: {info.device_name}")
    return "\n".join(lines)


def print_run_header(config: Any, paths: ProjectPaths, seed: Optional[int]) -> None:
    """Print a compact run banner before long training starts."""

    print("\n=== D-CDM Run ===")
    print(f"Project root: {paths.project_root}")
    print(f"Graph dir: {paths.graph_dir}")
    print(f"Output dir: {paths.output_dir}")
    print(f"Seed: {seed if seed is not None else 'not fixed'}")
    print(format_config(config))
    print(format_runtime_info(collect_runtime_info()))


def resolve_user_path(path: str | Path, base: Optional[Path] = None) -> Path:
    """Resolve a user-supplied path relative to a base directory."""

    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    if base is None:
        base = Path.cwd()
    return (base / raw).resolve()


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable with common truthy values."""

    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
