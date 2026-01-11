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


def get_demucs_model(model_name: str = 'mdx'):
    """
    Load and cache the Demucs model by name.

    Pre-installed models: mdx (default), htdemucs, mdx_extra

    Args:
        model_name: Name of the Demucs model to load

    Returns:
        Loaded Demucs model

    Raises:
        ValueError: If model_name is not recognized
        RuntimeError: If model fails to load
    """
    global _demucs_models

    # List of models pre-downloaded during Docker build
    PREINSTALLED_MODELS = ['mdx', 'htdemucs', 'mdx_extra']

    if model_name not in _demucs_models:
        try:
            device = get_device()

            # Provide helpful message for non-preinstalled models
            if model_name not in PREINSTALLED_MODELS:
                print(f"Warning: Model '{model_name}' is not pre-installed. Attempting to download...")

            model = get_model(name=model_name)
            model.to(device)
            model.eval()
            _demucs_models[model_name] = model

        except Exception as e:
            error_msg = f"Failed to load Demucs model '{model_name}': {str(e)}"
            if model_name not in PREINSTALLED_MODELS:
                error_msg += f"\nAvailable pre-installed models: {', '.join(PREINSTALLED_MODELS)}"
            raise RuntimeError(error_msg) from e

    return _demucs_models[model_name]
