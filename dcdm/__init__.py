"""Project package for the D-CDM ISAC graph diffusion experiment."""

from .config import Config, config

__all__ = [
    "Config",
    "config",
    "ISACGraphData",
    "DiscreteDiffusion",
    "compute_transition_powers",
    "reverse_step_digress",
    "DenoisingNetwork",
    "GraphTransformerLayer",
    "DCDM_Trainer",
    "DCDMTrainer",
    "TrainingHistory",
    "RewardSummary",
    "EpochRecord",
]


def __getattr__(name):
    if name == "ISACGraphData":
        from .data import ISACGraphData

        return ISACGraphData
    if name in {"DiscreteDiffusion", "compute_transition_powers", "reverse_step_digress"}:
        from . import diffusion

        return getattr(diffusion, name)
    if name in {"DenoisingNetwork", "GraphTransformerLayer"}:
        from . import model

        return getattr(model, name)
    if name in {"DCDM_Trainer", "DCDMTrainer"}:
        from . import trainer

        return getattr(trainer, name)
    if name in {"TrainingHistory", "RewardSummary", "EpochRecord"}:
        from . import metrics

        return getattr(metrics, name)
    raise AttributeError(f"module 'dcdm' has no attribute {name!r}")
