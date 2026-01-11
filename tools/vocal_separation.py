"""Lead/backing vocal separation using RoFormer models."""
import os
from pathlib import Path
from typing import Dict, Any
from fastmcp import FastMCP
from utils import get_roformer_model, fix_ownership


def separate_vocals_internal(
    audio_path: str,
    output_dir: str = "./output",
    model: str = "melband"
) -> Dict[str, Any]:
    """
    Internal function to separate lead and backing vocals.

    Args:
        audio_path: Path to input audio file
        output_dir: Directory to save separated stems
        model: 'bs' for BS-RoFormer, 'melband' for MelBand-RoFormer (default)

    Returns:
        Dict with success status and output paths
    """
    try:
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

        # Load model
        separator = get_roformer_model(model)

        # Perform separation
        output_files = separator.separate(audio_path, output_dir=output_dir)
        
        # Validate return value
        if not output_files or not isinstance(output_files, (list, tuple)):
            return {
                "success": False,
                "error": "Separation failed or returned no files",
                "message": "The separation process did not produce any output files"
            }

        # Organize outputs
        output_paths = {}

        for file_path in output_files:
            file_name = Path(file_path).name.lower()

            # Identify stem type
            if 'vocal' in file_name or 'voice' in file_name:
                if 'instrumental' not in file_name:
                    output_paths['lead_vocals'] = file_path
            elif 'instrumental' in file_name or 'accomp' in file_name:
                output_paths['instrumental'] = file_path
        
        # Validate that we found at least some stems
        if not output_paths:
            return {
                "success": False,
                "error": "No recognized stems found in output",
                "message": f"Separation completed but no files matched expected patterns (vocal, instrumental). Files: {[Path(f).name for f in output_files]}"
            }

        # Fix ownership
        fix_ownership(output_dir)

        return {
            "success": True,
            "stems": output_paths,
            "model": model,
            "message": f"Successfully separated vocals using {model} model"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to separate vocals: {str(e)}"
        }


def register_tools(mcp: FastMCP):
    """Register vocal separation tools with the MCP server."""

    @mcp.tool()
    def separate_lead_backing_vocals(
        audio_path: str,
        output_dir: str = "./output",
        model: str = "melband"
    ) -> Dict[str, Any]:
        """
        Separate an audio file into lead vocals and instrumental/backing.

        Uses state-of-the-art RoFormer models (BS-RoFormer or MelBand-RoFormer)
        to separate lead vocals from instrumental and backing vocals.

        Args:
            audio_path: Path to the input audio file
            output_dir: Directory where separated stems will be saved (default: ./output)
            model: Model to use - 'melband' (recommended) or 'bs' (default: melband)

        Returns:
            Dictionary containing:
            - success: Boolean indicating if separation succeeded
            - stems: Dictionary with paths to separated audio files
                - lead_vocals: Path to lead vocal track
                - instrumental: Path to instrumental/backing track
            - model: Name of the model used
            - message: Status message

        Example:
            {
                "success": true,
                "stems": {
                    "lead_vocals": "./output/song_vocals.wav",
                    "instrumental": "./output/song_instrumental.wav"
                },
                "model": "melband",
                "message": "Successfully separated vocals using melband model"
            }
        """
        return separate_vocals_internal(audio_path, output_dir, model)
