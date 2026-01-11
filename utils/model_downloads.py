"""Utilities for downloading and managing model weights."""
import os
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


def download_satb_models():
    """Download SATB Conditioned U-Net models from Google Drive."""
    import gdown

    # Google Drive file IDs from research paper folder
    # https://drive.google.com/drive/folders/1zqdSLCGJ7cqw7oCP6iEhh3t2uIY9sC31
    # Models are Keras/HDF5 format (.h5)
    models = {
        'c-unet_ds_l.h5': '1X-Suj1uDxAkmArT8qMN5z8cRI7BjDhub',  # Local model (115.3 MB)
        'c-unet_ds_g.h5': '1e0zHaGRPXKlIIc6caNiqDcZmDdrAkU-m',  # Global model (118.3 MB)
    }

    satb_dir = MODEL_DIR / "satb"
    satb_dir.mkdir(parents=True, exist_ok=True)

    for model_name, file_id in models.items():
        model_path = satb_dir / model_name
        if model_path.exists():
            print(f"✓ {model_name} already exists", flush=True)
            continue

        print(f"Downloading {model_name} from Google Drive...", flush=True)
        try:
            # Download from Google Drive using file ID
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(model_path), quiet=False, fuzzy=True)
            print(f"✓ {model_name} downloaded successfully", flush=True)
        except Exception as e:
            print(f"✗ Failed to download {model_name}: {e}", flush=True)
            print(f"Manual download: https://drive.google.com/file/d/{file_id}/view", flush=True)
            print(f"Or visit: https://drive.google.com/drive/folders/1zqdSLCGJ7cqw7oCP6iEhh3t2uIY9sC31", flush=True)

    print(f"SATB models directory: {satb_dir}", flush=True)
