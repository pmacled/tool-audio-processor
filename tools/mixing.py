"""
Audio mixing tools including replace_layer and mix_layers.
"""

import os
import tempfile
import shutil
from typing import List, Dict, Any, Optional

import librosa
import numpy as np
import soundfile as sf
from fastmcp import FastMCP

from utils import fix_ownership


def register_tools(mcp: FastMCP):
    """Register mixing tools with the MCP server."""
    
    @mcp.tool()
    def replace_layer(
        original_mix_path: str,
        layer_to_replace: str,
        new_layer_path: str,
        output_path: str = "./output/replaced_mix.wav"
    ) -> Dict[str, Any]:
        """
        Replace a specific layer in a mixed audio file with a new layer.
        
        Args:
            original_mix_path: Path to the original mixed audio
            layer_to_replace: Layer to replace (vocals, drums, bass, or other)
            new_layer_path: Path to the new layer audio file
            output_path: Path to save the output mix
        
        Returns:
            Dictionary with path to output mix and metadata
        """
        try:
            # Import separation function (public API)
            from tools.separation import separate_audio_internal
            
            # First, separate the original mix
            temp_dir = tempfile.mkdtemp()
            
            try:
                separation_result = separate_audio_internal(original_mix_path, temp_dir)
                
                if not separation_result["success"]:
                    return separation_result
                
                layers = separation_result["layers"]
                
                # Validate layer name
                if layer_to_replace not in layers:
                    return {
                        "success": False,
                        "error": f"Invalid layer name: {layer_to_replace}",
                        "message": f"Layer must be one of: {', '.join(layers.keys())}"
                    }
                
                # Determine reference sample rate from separated layers (original mix)
                # Use the first layer's native sample rate as the target for all layers.
                first_layer_path = next(iter(layers.values()))
                _, target_sr = librosa.load(first_layer_path, sr=None)
                
                # Load all layers
                layer_audio = {}
                max_length = 0
                
                for name, path in layers.items():
                    if name == layer_to_replace:
                        # Load the new layer, resampling to match the original mix's sample rate
                        audio, _ = librosa.load(new_layer_path, sr=target_sr)
                    else:
                        # Load existing layers, resampling (if needed) to the target sample rate
                        audio, _ = librosa.load(path, sr=target_sr)
                    
                    layer_audio[name] = audio
                    max_length = max(max_length, len(audio))
                
                # Use the target sample rate for subsequent processing and saving
                sr = target_sr
                
                # Pad all layers to the same length
                for name in layer_audio:
                    if len(layer_audio[name]) < max_length:
                        layer_audio[name] = np.pad(
                            layer_audio[name], 
                            (0, max_length - len(layer_audio[name]))
                        )
                
                # Mix all layers
                mixed = np.zeros(max_length)
                num_layers = len(layer_audio)
                if num_layers == 0:
                    return {
                        "success": False,
                        "error": "No layers to mix",
                        "message": "No audio layers were loaded"
                    }
                
                for audio in layer_audio.values():
                    mixed += audio
                
                # Normalize
                mixed = mixed / num_layers
                
                # Ensure output directory exists (if a directory is specified)
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)

                # Save output
                sf.write(output_path, mixed, sr)

                # Fix ownership of output file and directory
                fix_ownership(output_path)
                if output_dir:
                    fix_ownership(output_dir)

                return {
                    "success": True,
                    "output_path": output_path,
                    "replaced_layer": layer_to_replace,
                    "sample_rate": sr,
                    "duration": float(max_length / sr),
                    "message": f"Successfully replaced {layer_to_replace} layer"
                }
            finally:
                # Clean up temporary directory
                shutil.rmtree(temp_dir, ignore_errors=True)
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to replace layer: {str(e)}"
            }
    
    @mcp.tool()
    def mix_layers(
        layer_paths: List[str],
        output_path: str = "./output/mixed.wav",
        layer_volumes: Optional[List[float]] = None,
        normalize_output: bool = True
    ) -> Dict[str, Any]:
        """
        Combine multiple audio layers into a single mixed output.
        
        Args:
            layer_paths: List of paths to audio files to mix
            output_path: Path to save mixed audio
            layer_volumes: Optional list of volume multipliers for each layer (default: equal volumes)
            normalize_output: Whether to normalize the output (default: True)
        
        Returns:
            Dictionary with path to mixed audio and metadata
        """
        try:
            if not layer_paths:
                return {
                    "success": False,
                    "error": "No layer paths provided",
                    "message": "At least one layer path is required"
                }
            
            # Set default volumes if not provided
            if layer_volumes is None:
                layer_volumes = [1.0] * len(layer_paths)
            elif len(layer_volumes) != len(layer_paths):
                return {
                    "success": False,
                    "error": "Volume list length must match layer paths length",
                    "message": f"Expected {len(layer_paths)} volumes, got {len(layer_volumes)}"
                }
            
            # Load all layers
            layers = []
            sr = None
            max_length = 0
            
            for path in layer_paths:
                audio, sr = librosa.load(path, sr=sr)
                layers.append(audio)
                max_length = max(max_length, len(audio))
            
            # Pad all layers to the same length
            for i in range(len(layers)):
                if len(layers[i]) < max_length:
                    layers[i] = np.pad(layers[i], (0, max_length - len(layers[i])))
            
            # Mix layers with volumes
            mixed = np.zeros(max_length)
            for i, (layer, volume) in enumerate(zip(layers, layer_volumes)):
                mixed += layer * volume
            
            # Normalize if requested
            if normalize_output:
                max_val = np.max(np.abs(mixed))
                if max_val > 0:
                    mixed = mixed / max_val

            # Ensure output directory exists (handle case where output_path has no directory component)
            output_dir = os.path.dirname(output_path) or "."
            os.makedirs(output_dir, exist_ok=True)

            # Save output
            sf.write(output_path, mixed, sr)

            # Fix ownership of output file and directory
            fix_ownership(output_path)
            if output_dir != ".":
                fix_ownership(output_dir)

            return {
                "success": True,
                "output_path": output_path,
                "num_layers": len(layer_paths),
                "layer_volumes": layer_volumes,
                "sample_rate": sr,
                "duration": float(max_length / sr),
                "normalized": normalize_output,
                "message": f"Successfully mixed {len(layer_paths)} layers"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to mix layers: {str(e)}"
            }
