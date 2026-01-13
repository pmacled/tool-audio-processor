# Model Weights Directory

This directory contains pre-downloaded model weights for vocal separation.

## Directory Structure
- `roformer/` - BS-RoFormer and MelBand-RoFormer models (auto-downloaded)

## Models

### RoFormer Models (Automatic)
Downloaded automatically via `audio-separator` during Docker build:
- `model_bs_roformer_ep_317_sdr_12.9755.ckpt` - BS-RoFormer variant
- `model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt` - MelBand-RoFormer variant

**Framework:** TensorFlow/Keras (not PyTorch)
**Research:** UPF Barcelona, ISMIR 2020
**Source:** https://drive.google.com/drive/folders/1zqdSLCGJ7cqw7oCP6iEhh3t2uIY9sC31
