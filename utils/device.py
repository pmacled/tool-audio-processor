"""
Device and model management utilities.
"""

import torch
from pathlib import Path
from demucs.pretrained import get_model

# Global caches
_demucs_models = {}
_roformer_models = {}
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
        ValueError: If model_name is invalid (empty, None, or not a string)
        RuntimeError: If model fails to load
    """
    global _demucs_models

    # Validate model_name parameter - type check first
    if not isinstance(model_name, str):
        raise ValueError(
            f"Invalid model_name: must be a string, got {type(model_name).__name__}: {repr(model_name)}"
        )
    
    # Strip whitespace to avoid silent failures
    model_name = model_name.strip()
    if not model_name:
        raise ValueError("Invalid model_name: cannot be empty or whitespace-only")

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


def get_roformer_model(model_name: str = 'melband', output_dir: str = './output'):
    """
    Load RoFormer models (BS-RoFormer or MelBand-RoFormer).

    Note: Unlike Demucs and SATB models, this function does NOT cache the Separator
    instance because audio-separator requires output_dir to be specified during
    initialization. The audio-separator library caches model weights internally,
    so creating a new Separator instance for each call is still efficient.

    Pre-installed models: melband (default), bs

    Args:
        model_name: 'bs' for BS-RoFormer, 'melband' for MelBand-RoFormer
        output_dir: Directory where separated files will be saved

    Returns:
        Separator instance with loaded model

    Raises:
        ValueError: If model_name is invalid
        RuntimeError: If model fails to load
    """
    # Validation
    if not isinstance(model_name, str):
        raise ValueError(
            f"Invalid model_name: must be a string, got {type(model_name).__name__}: {repr(model_name)}"
        )

    model_name = model_name.strip().lower()
    if not model_name:
        raise ValueError("Invalid model_name: cannot be empty or whitespace-only")

    # Model mapping
    MODEL_MAP = {
        'bs': 'model_bs_roformer_ep_317_sdr_12.9755.ckpt',
        'melband': 'model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt',
    }

    if model_name not in MODEL_MAP:
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {list(MODEL_MAP.keys())}"
        )

    model_filename = MODEL_MAP[model_name]

    try:
        from audio_separator.separator import Separator

        # Use absolute path to container's models directory
        # This ensures we use pre-downloaded models even when working directory is /workspace
        model_dir = Path("/app/models/roformer")

        # Create Separator with output_dir - required by audio-separator API
        separator = Separator(
            model_file_dir=str(model_dir),
            output_dir=output_dir,
            output_format='wav'
        )
        separator.load_model(model_filename=model_filename)
        return separator

    except Exception as e:
           raise RuntimeError(
                f"Failed to load RoFormer model '{model_name}': {str(e)}"
            ) from e

    return _roformer_models[model_name]
