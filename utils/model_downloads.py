"""Utilities for downloading and managing model weights."""
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models"


def download_roformer_models():
    """Pre-download RoFormer models during build."""
    from audio_separator.separator import Separator

    models = [
        'model_bs_roformer_ep_317_sdr_12.9755.ckpt',  # BS-RoFormer
        'model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt',  # MelBand-RoFormer
    ]

    roformer_dir = MODEL_DIR / "roformer"
    roformer_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        print(f"Downloading {model}...", flush=True)
        try:
            separator = Separator(model_file_dir=str(roformer_dir))
            separator.load_model(model_filename=model)
            print(f"✓ {model} downloaded successfully", flush=True)
        except Exception as e:
            print(f"✗ Failed to download {model}: {e}", flush=True)
            raise

