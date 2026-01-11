"""
Device and model management utilities.
"""

import torch
from demucs.pretrained import get_model

# Global model cache
_demucs_model = None
_device = None


def get_device():
    """Get the appropriate device (CUDA if available, otherwise CPU)."""
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def get_demucs_model(model_name: str = 'htdemucs'):
    """Load and cache the Demucs model."""
    global _demucs_model
    if _demucs_model is None:
        device = get_device()
        _demucs_model = get_model(name=model_name)
        _demucs_model.to(device)
        _demucs_model.eval()
    return _demucs_model
