"""
Audio separation tool using Demucs.
"""

import os
from pathlib import Path
from typing import Dict, Any

import torch
import torchaudio
from demucs.apply import apply_model
from demucs.audio import save_audio
from fastmcp import FastMCP

from utils import get_device, get_demucs_model, fix_ownership


def _separate_audio_internal(audio_path: str, output_dir: str, model: str = "htdemucs") -> Dict[str, Any]:
    """Internal function to separate audio, used by other tools."""
    try:
        # Validate input file exists
        if not os.path.isfile(audio_path):
            return {
                "success": False,
                "error": f"Audio file not found: {audio_path}",
                "message": f"The specified audio file does not exist: {audio_path}"
            }
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        device = get_device()
        
        # Load model (use the specified model parameter)
        demucs_model = get_demucs_model(model)
        
        # Apply separation
        audio = audio.to(device)
        
        # Demucs expects specific sample rate
        if sr != demucs_model.samplerate:
            audio = torchaudio.functional.resample(
                audio, 
                orig_freq=sr, 
                new_freq=demucs_model.samplerate
            )
            sr = demucs_model.samplerate
        
        # Apply model
        with torch.no_grad():
            sources = apply_model(demucs_model, audio[None], device=device)[0]
        
        # Save separated sources
        sources = sources.cpu()
        source_names = ['drums', 'bass', 'other', 'vocals']
        output_paths = {}
        
        for i, name in enumerate(source_names):
            output_path = os.path.join(output_dir, f"{Path(audio_path).stem}_{name}.wav")
            save_audio(sources[i], output_path, sr, clip='clamp', as_float=False)
            output_paths[name] = output_path

        # Fix ownership of output directory and all created files
        fix_ownership(output_dir)

        return {
            "success": True,
            "layers": output_paths,
            "sample_rate": sr,
            "device": str(device),
            "message": f"Successfully separated audio into {len(source_names)} layers"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to separate audio: {str(e)}"
        }


def register_tools(mcp: FastMCP):
    """Register separation tools with the MCP server."""
    
    @mcp.tool()
    def separate_audio_layers(
        audio_path: str,
        output_dir: str = "./output",
        model: str = "htdemucs"
    ) -> Dict[str, Any]:
        """
        Separate audio into vocals, drums, bass, and other layers using Demucs.
        
        Args:
            audio_path: Path to the input audio file
            output_dir: Directory to save separated layers (default: ./output)
            model: Demucs model to use (default: htdemucs)
        
        Returns:
            Dictionary with paths to separated layer files and metadata
        """
        return _separate_audio_internal(audio_path, output_dir, model)
