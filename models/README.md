# Model Weights Directory

This directory contains pre-downloaded model weights for vocal separation.

## Directory Structure
- `roformer/` - BS-RoFormer and MelBand-RoFormer models (auto-downloaded)
- `satb/` - SATB Conditioned U-Net models (auto-downloaded from Google Drive)

## Models

### RoFormer Models (Automatic)
Downloaded automatically via `audio-separator` during Docker build:
- `model_bs_roformer_ep_317_sdr_12.9755.ckpt` - BS-RoFormer variant
- `model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt` - MelBand-RoFormer variant

### SATB Models (Automatic via gdown)
Downloaded automatically from Google Drive during Docker build:
- `c-unet_ds_l.h5` - Local domain conditioning (115.3 MB, Keras/HDF5 format)
- `c-unet_ds_g.h5` - Global domain conditioning (118.3 MB, Keras/HDF5 format)

**Framework:** TensorFlow/Keras (not PyTorch)
**Research:** UPF Barcelona, ISMIR 2020
**Source:** https://drive.google.com/drive/folders/1zqdSLCGJ7cqw7oCP6iEhh3t2uIY9sC31

## Manual Download (Fallback)
If automated download fails during build, manually download SATB models:
1. Visit: https://drive.google.com/drive/folders/1zqdSLCGJ7cqw7oCP6iEhh3t2uIY9sC31
2. Download both model files:
   - `c-unet_ds_l.h5` (Local model)
   - `c-unet_ds_g.h5` (Global model)
3. Place in `satb/` directory
