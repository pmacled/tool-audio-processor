"""
MIDI synthesis tool.
"""

import os
from typing import Dict, Any

import numpy as np
import pretty_midi
import soundfile as sf
from fastmcp import FastMCP

from utils import fix_ownership


def register_tools(mcp: FastMCP):
    """Register synthesis tools with the MCP server."""
    
    @mcp.tool()
    def synthesize_instrument_layer(
        midi_path: str,
        instrument: str = "piano",
        output_path: str = "./output/synthesized.wav",
        sample_rate: int = 44100
    ) -> Dict[str, Any]:
        """
        Generate a new instrument layer from MIDI data.
        
        Args:
            midi_path: Path to MIDI file
            instrument: Instrument type to synthesize (default: piano)
            output_path: Path to save synthesized audio
            sample_rate: Sample rate for output audio (default: 44100)
        
        Returns:
            Dictionary with path to synthesized audio and metadata
        """
        try:
            # Load MIDI file using pretty_midi
            midi_data = pretty_midi.PrettyMIDI(midi_path)

            # Optionally override instrument programs based on requested instrument
            if instrument:
                instrument_program_map = {
                    "piano": 0,                 # Acoustic Grand Piano
                    "acoustic grand piano": 0,
                    "bright piano": 1,          # Bright Acoustic Piano
                    "electric piano": 4,        # Electric Piano 1
                    "e-piano": 4,
                    "organ": 16,                # Drawbar Organ
                    "guitar": 24,               # Nylon Acoustic Guitar
                    "acoustic guitar": 24,
                    "electric guitar": 27,      # Electric Guitar (clean)
                    "violin": 40,
                    "viola": 41,
                    "cello": 42,
                    "bass": 32,                 # Acoustic Bass
                    "synth bass": 38,           # Synth Bass 1
                    "flute": 73,
                    "sax": 65,                  # Alto Sax
                    "trumpet": 56,
                }
                program = instrument_program_map.get(str(instrument).lower())
                if program is not None:
                    for inst in midi_data.instruments:
                        inst.program = program
                        # Ensure we treat this as a pitched instrument, not percussion
                        inst.is_drum = False
            
            # Synthesize audio
            audio = midi_data.fluidsynth(fs=sample_rate)
            
            # Normalize audio (avoid division by zero for silent audio)
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            
            # Ensure output directory exists (if a directory is specified)
            dir_name = os.path.dirname(output_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            # Save audio
            sf.write(output_path, audio, sample_rate)

            # Fix ownership of output file and directory
            fix_ownership(output_path)
            if dir_name:
                fix_ownership(dir_name)

            # Get MIDI info
            total_notes = sum(len(inst.notes) for inst in midi_data.instruments)
            
            return {
                "success": True,
                "output_path": output_path,
                "instrument": instrument,
                "sample_rate": sample_rate,
                "duration": float(midi_data.get_end_time()),
                "total_notes": total_notes,
                "instruments_in_midi": len(midi_data.instruments),
                "message": f"Successfully synthesized {instrument} from MIDI"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to synthesize instrument: {str(e)}"
            }
