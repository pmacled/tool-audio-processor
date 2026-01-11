"""SATB (Soprano/Alto/Tenor/Bass) voice part separation using Conditioned U-Net."""
import os
from pathlib import Path
from typing import Dict, Any
import numpy as np
import soundfile as sf
import librosa
from fastmcp import FastMCP
from utils import get_satb_model, fix_ownership


def separate_satb_internal(
    audio_path: str,
    output_dir: str = "./output",
    model_type: str = "local"
) -> Dict[str, Any]:
    """
    Internal function to separate SATB voice parts.

    Args:
        audio_path: Path to choral audio file
        output_dir: Directory to save separated voice parts
        model_type: 'local' or 'global' domain conditioning

    Returns:
        Dict with success status and output paths
    """
    try:
        import tensorflow as tf

        # Validation
        if not os.path.isfile(audio_path):
            return {
                "success": False,
                "error": f"Audio file not found: {audio_path}",
                "message": f"The specified audio file does not exist: {audio_path}"
            }

        # Validate and sanitize output_dir to prevent path traversal
        output_dir = os.path.abspath(output_dir)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Load audio (model expects 22050 Hz)
        target_sr = 22050
        print(f"Loading audio from {audio_path}...", flush=True)
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=False)

        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)

        print(f"Audio loaded: shape={audio.shape}, sr={sr}", flush=True)

        # Load model
        model = get_satb_model(model_type)

        # STFT parameters
        # These must match the parameters used during model training.
        # The ISMIR 2020 paper uses standard U-Net STFT settings.
        n_fft = 2048
        hop_length = 512

        # Compute STFT
        print("Computing STFT...", flush=True)
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Normalize magnitude for model input
        # Note: For silent audio (all zeros), this will produce all zeros
        # which is the expected behavior (silence in = silence out)
        magnitude_normalized = magnitude / (np.max(magnitude) + 1e-8)

        # Prepare input for model: (batch, height, width, channels)
        # Model expects (batch, freq_bins, time_frames, 1)
        magnitude_batch = magnitude_normalized.T[np.newaxis, :, :, np.newaxis]  # Add batch and channel dims

        print(f"Model input shape: {magnitude_batch.shape}", flush=True)

        # Separate each voice part using conditioning
        voice_parts = ['soprano', 'alto', 'tenor', 'bass']
        output_paths = {}
        base_name = Path(audio_path).stem

        for voice_id, voice_name in enumerate(voice_parts):
            print(f"Separating {voice_name}...", flush=True)

            # Create one-hot condition for this voice part
            condition = np.zeros((1, 4), dtype=np.float32)
            condition[0, voice_id] = 1.0

            # Run model prediction
            # The ISMIR 2020 Conditioned U-Net is trained with two inputs:
            # [spectrogram, one-hot voice-condition] for voice part separation.
            try:
                # Predict using conditioned inputs; this is required for distinct SATB outputs.
                predicted_mask = model.predict([magnitude_batch, condition], verbose=0)
            except Exception as e:
                # Do not fall back to an unconditioned prediction, as that would produce
                # identical outputs for all voice parts and defeat SATB separation.
                raise RuntimeError(
                    f"Conditioned SATB prediction failed for voice '{voice_name}': {e}"
                )

            # Remove batch and channel dimensions
            predicted_mask = np.squeeze(predicted_mask)

            # Ensure mask shape matches magnitude spectrogram
            if predicted_mask.shape != magnitude_normalized.shape:
                # Try simple transpose if it fixes the mismatch
                if predicted_mask.T.shape == magnitude_normalized.shape:
                    predicted_mask = predicted_mask.T
                else:
                    raise ValueError(
                        f"Predicted mask shape {predicted_mask.shape} (or its transpose "
                        f"{predicted_mask.T.shape}) does not match magnitude shape "
                        f"{magnitude_normalized.shape} for voice part '{voice_name}'."
                    )

            # Apply mask to magnitude spectrogram
            separated_magnitude = magnitude * predicted_mask

            # Reconstruct complex spectrogram with original phase
            separated_stft = separated_magnitude * np.exp(1j * phase)

            # Inverse STFT
            separated_audio = librosa.istft(separated_stft, hop_length=hop_length, length=len(audio))

            # Normalize audio to prevent clipping
            # Note: For silent audio segments, this will produce silence (expected behavior)
            separated_audio = separated_audio / (np.max(np.abs(separated_audio)) + 1e-8) * 0.9

            # Save separated voice
            output_path = os.path.join(output_dir, f"{base_name}_{voice_name}.wav")
            sf.write(output_path, separated_audio, sr)
            output_paths[voice_name] = output_path
            print(f"✓ {voice_name} saved to {output_path}", flush=True)

        # Fix ownership
        fix_ownership(output_dir)

        return {
            "success": True,
            "voice_parts": output_paths,
            "sample_rate": sr,
            "model_type": model_type,
            "message": f"Successfully separated SATB voice parts using {model_type} model"
        }

    except FileNotFoundError as e:
        return {
            "success": False,
            "error": str(e),
            "message": str(e)
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return {
            "success": False,
            "error": str(e),
            "traceback": error_trace,
            "message": f"Failed to separate SATB voice parts: {str(e)}"
        }


def register_tools(mcp: FastMCP):
    """Register SATB separation tools with the MCP server."""

    @mcp.tool()
    def separate_satb_voices(
        audio_path: str,
        output_dir: str = "./output",
        model_type: str = "local"
    ) -> Dict[str, Any]:
        """
        Separate a choral audio file into SATB (Soprano/Alto/Tenor/Bass) voice parts.

        Uses a Conditioned U-Net model specifically trained for choral music
        to separate the four standard voice parts: Soprano, Alto, Tenor, and Bass.
        Models are based on research from UPF Barcelona (ISMIR 2020).

        Args:
            audio_path: Path to the input choral audio file
            output_dir: Directory where separated voice parts will be saved (default: ./output)
            model_type: Conditioning type - 'local' (default) or 'global'

        Returns:
            Dictionary containing:
            - success: Boolean indicating if separation succeeded
            - voice_parts: Dictionary with paths to separated voice files
                - soprano: Path to soprano voice track
                - alto: Path to alto voice track
                - tenor: Path to tenor voice track
                - bass: Path to bass voice track
            - sample_rate: Sample rate of output audio (22050 Hz)
            - model_type: Type of conditioning used
            - message: Status message

        Example:
            {
                "success": true,
                "voice_parts": {
                    "soprano": "./output/choir_soprano.wav",
                    "alto": "./output/choir_alto.wav",
                    "tenor": "./output/choir_tenor.wav",
                    "bass": "./output/choir_bass.wav"
                },
                "sample_rate": 22050,
                "model_type": "local",
                "message": "Successfully separated SATB voice parts using local model"
            }
        """
        return separate_satb_internal(audio_path, output_dir, model_type)
