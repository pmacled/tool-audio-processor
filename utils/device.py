"""
Device and model management utilities.
"""

import torch
from demucs.pretrained import get_model

# Global caches
_demucs_models = {}
_device = None


def get_device():
    """Get the appropriate device (CUDA if available, otherwise CPU)."""
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def get_demucs_model(model_name: str = 'htdemucs'):
    """Load and cache the Demucs model by name."""
    global _demucs_models
    if model_name not in _demucs_models:
        device = get_device()
        model = get_model(name=model_name)
        model.to(device)
        model.eval()
        _demucs_models[model_name] = model
    return _demucs_models[model_name]
