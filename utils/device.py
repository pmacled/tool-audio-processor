"""
Device and model management utilities.
"""

import torch
from pathlib import Path
from demucs.pretrained import get_model

# Global caches
_demucs_models = {}
_roformer_models = {}
_satb_models = {}
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


def get_roformer_model(model_name: str = 'melband'):
    """
    Load and cache RoFormer models (BS-RoFormer or MelBand-RoFormer).

    Pre-installed models: melband (default), bs

    Args:
        model_name: 'bs' for BS-RoFormer, 'melband' for MelBand-RoFormer

    Returns:
        Separator instance with loaded model

    Raises:
        ValueError: If model_name is invalid
        RuntimeError: If model fails to load
    """
    global _roformer_models

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

    # Cache check
    if model_name not in _roformer_models:
        try:
            from audio_separator.separator import Separator

            device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_dir = Path(__file__).parent.parent / "models" / "roformer"

            separator = Separator(
                model_filename=model_filename,
                model_file_dir=str(model_dir),
                output_format='wav',
                device=device_name
            )
            separator.load_model()
            _roformer_models[model_name] = separator

        except Exception as e:
            raise RuntimeError(
                f"Failed to load RoFormer model '{model_name}': {str(e)}"
            ) from e

    return _roformer_models[model_name]


def get_satb_model(model_type: str = 'local'):
    """
    Load and cache SATB Conditioned U-Net Keras model.

    Args:
        model_type: 'local' or 'global' domain conditioning

    Returns:
        Loaded Keras SATB model

    Raises:
        ValueError: If model_type is invalid
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model fails to load
    """
    global _satb_models

    # Validation
    if not isinstance(model_type, str):
        raise ValueError(
            f"Invalid model_type: must be a string, got {type(model_type).__name__}"
        )

    model_type = model_type.strip().lower()
    if model_type not in ['local', 'global']:
        raise ValueError(
            f"Invalid model_type: {model_type}. Must be 'local' or 'global'"
        )

    # Cache check
    if model_type not in _satb_models:
        try:
            import tensorflow as tf
            from tensorflow import keras

            # Configure TensorFlow to use GPU if available
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Enable memory growth to avoid allocating all GPU memory
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(f"GPU configuration warning: {e}", flush=True)

            # Model filename mapping
            model_files = {
                'local': 'c-unet_ds_l.h5',
                'global': 'c-unet_ds_g.h5'
            }

            model_path = Path(__file__).parent.parent / "models" / "satb" / model_files[model_type]

            if not model_path.exists():
                raise FileNotFoundError(
                    f"SATB model not found at {model_path}. "
                    f"Please download from: https://drive.google.com/drive/folders/1zqdSLCGJ7cqw7oCP6iEhh3t2uIY9sC31"
                )

            # Load Keras model
            print(f"Loading SATB {model_type} model from {model_path}...", flush=True)
            model = keras.models.load_model(str(model_path), compile=False)
            _satb_models[model_type] = model
            print(f"✓ SATB {model_type} model loaded successfully", flush=True)

        except Exception as e:
            raise RuntimeError(
                f"Failed to load SATB model '{model_type}': {str(e)}"
            ) from e

    return _satb_models[model_type]
