"""
Audio effects tool.
"""

import os
from typing import Dict, Any, Optional

import librosa
import numpy as np
import soundfile as sf
from fastmcp import FastMCP

from utils import fix_ownership


def register_tools(mcp: FastMCP):
    """Register effects tools with the MCP server."""
    
    @mcp.tool()
    def modify_layer(
        audio_path: str,
        effect: str,
        output_path: str = "./output/modified.wav",
        steps: Optional[int] = 0,
        rate: Optional[float] = 1.0,
        target_db: Optional[float] = -20.0,
        fade_in: Optional[float] = 0.0,
        fade_out: Optional[float] = 0.0,
        decay: Optional[float] = 0.5
    ) -> Dict[str, Any]:
        """
        Apply audio effects to a layer.

        Args:
            audio_path: Path to the audio file to modify
            effect: Effect to apply - "pitch_shift", "time_stretch", "reverb", "normalize", "fade"
            output_path: Path to save modified audio
            steps: Semitones to shift pitch (for pitch_shift effect, default: 0)
            rate: Speed multiplier (for time_stretch effect, default: 1.0)
            target_db: Target dB level (for normalize effect, default: -20.0)
            fade_in: Fade in duration in seconds (for fade effect, default: 0.0)
            fade_out: Fade out duration in seconds (for fade effect, default: 0.0)
            decay: Reverb decay amount (for reverb effect, default: 0.5)

        Returns:
            Dictionary with path to modified audio and metadata
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)

            modified = y.copy()
            applied_params = {}

            # Apply effects
            if effect == "pitch_shift":
                modified = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
                applied_params["steps"] = steps

            elif effect == "time_stretch":
                modified = librosa.effects.time_stretch(y, rate=rate)
                applied_params["rate"] = rate

            elif effect == "normalize":
                # Calculate current RMS
                rms = np.sqrt(np.mean(modified**2))
                current_db = 20 * np.log10(rms) if rms > 0 else -100
                # Calculate gain needed
                gain_db = target_db - current_db
                gain = 10 ** (gain_db / 20)
                modified = modified * gain
                applied_params["target_db"] = target_db
                applied_params["gain_applied"] = float(gain)

            elif effect == "fade":
                fade_in_samples = int(fade_in * sr)
                fade_out_samples = int(fade_out * sr)

                # Apply fade in
                if fade_in_samples > 0:
                    fade_in_curve = np.linspace(0, 1, fade_in_samples)
                    modified[:fade_in_samples] *= fade_in_curve

                # Apply fade out
                if fade_out_samples > 0:
                    fade_out_curve = np.linspace(1, 0, fade_out_samples)
                    modified[-fade_out_samples:] *= fade_out_curve

                applied_params["fade_in"] = fade_in
                applied_params["fade_out"] = fade_out

            elif effect == "reverb":
                # Simple reverb using convolution (simplified)
                # In production, you'd want a proper reverb implementation
                delay_samples = int(0.05 * sr)  # 50ms delay

                reverb_response = np.zeros(delay_samples)
                reverb_response[0] = 1.0
                reverb_response[-1] = decay

                modified = np.convolve(modified, reverb_response, mode='same')
                # Normalize reverb output to avoid harsh clipping from convolution overflows
                peak = np.max(np.abs(modified))
                if peak > 1.0:
                    modified = modified / peak
                    applied_params["reverb_peak_before_normalization"] = float(peak)
                applied_params["decay"] = decay
            else:
                return {
                    "success": False,
                    "error": f"Unknown effect: {effect}",
                    "message": "Effect must be one of: pitch_shift, time_stretch, reverb, normalize, fade"
                }
            
            # Clip to prevent distortion
            modified = np.clip(modified, -1.0, 1.0)

            # Ensure output directory exists (if a directory is specified)
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Save output
            sf.write(output_path, modified, sr)

            # Fix ownership of output file and directory
            fix_ownership(output_path)
            if output_dir:
                fix_ownership(output_dir)

            return {
                "success": True,
                "output_path": output_path,
                "effect": effect,
                "effect_params": applied_params,
                "sample_rate": sr,
                "duration": float(len(modified) / sr),
                "message": f"Successfully applied {effect} effect"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to modify layer: {str(e)}"
            }
