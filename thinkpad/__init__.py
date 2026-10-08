"""Think-Pad: a dual-stream transformer language model and its GPT baseline."""

from .config import MODEL_PRESETS, ModelConfig, TrainConfig
from .model import BaselineGPT, ThinkPadGPT, build_model, count_params, load_state_dict_compat

__all__ = [
    "MODEL_PRESETS",
    "ModelConfig",
    "TrainConfig",
    "BaselineGPT",
    "ThinkPadGPT",
    "build_model",
    "count_params",
    "load_state_dict_compat",
]
