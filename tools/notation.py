"""
Music notation export tool for converting MIDI to various notation formats.
"""

import os
from typing import Dict, Any, Optional

from fastmcp import FastMCP

from utils import fix_ownership


def register_tools(mcp: FastMCP):
    """Register notation export tools with the MCP server."""
    
    @mcp.tool()
    def export_notation(
        midi_path: str,
        output_path: str = "./output/notation.musicxml",
        output_format: str = "musicxml",
        title: Optional[str] = None,
        composer: Optional[str] = None,
        key_signature: Optional[str] = None,
        time_signature: str = "4/4",
        tempo: Optional[float] = None,
        clef: str = "treble"
    ) -> Dict[str, Any]:
        """
        Convert MIDI file to human-readable music notation formats.

        Args:
            midi_path: Path to MIDI file to convert
            output_path: Path to save notation file (default: ./output/notation.musicxml)
            output_format: Output format - "musicxml", "lilypond", "pdf", "png" (default: musicxml)
            title: Title of the piece (default: None)
            composer: Composer name (default: None)
            key_signature: Key signature, e.g., "C", "D#", "Bb" (default: None)
            time_signature: Time signature (default: 4/4)
            tempo: Tempo in BPM for display (default: None)
            clef: Clef type - "treble", "bass", "alto" (default: treble)

        Returns:
            Dictionary with output path, format, measure count, and metadata
        """
        try:
            # Validate input file exists
            if not os.path.isfile(midi_path):
                return {
                    "success": False,
                    "error": f"MIDI file not found: {midi_path}",
                    "message": f"The specified MIDI file does not exist: {midi_path}"
                }

            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            try:
                from music21 import converter, metadata, tempo as m21_tempo, key, meter, clef as m21_clef

                # Load MIDI file
                score = converter.parse(midi_path)

                # Add metadata
                if title or composer:
                    if not score.metadata:
                        score.metadata = metadata.Metadata()
                    if title:
                        score.metadata.title = title
                    if composer:
                        score.metadata.composer = composer

                # Add musical context to the first part
                if score.parts:
                    first_part = score.parts[0]

                    # Add elements in the correct order: clef, key signature, time signature, tempo
                    # Since we use insert(0, ...) which puts each element at the start,
                    # we add them in reverse order: tempo, time sig, key sig, clef
                    # This results in the final order: clef, key sig, time sig, tempo
                    
                    # Add tempo marking (inserted first in code, appears last in score)
                    if tempo:
                        tempo_mark = m21_tempo.MetronomeMark(number=tempo)
                        first_part.insert(0, tempo_mark)

                    # Add time signature
                    if time_signature:
                        ts = meter.TimeSignature(time_signature)
                        first_part.insert(0, ts)

                    # Add key signature
                    if key_signature:
                        try:
                            ks = key.Key(key_signature)
                            first_part.insert(0, ks)
                        except (ValueError, TypeError):
                            pass  # Invalid key signature, skip

                    # Set clef (inserted first, so it appears first)
                    clef_map = {
                        "treble": m21_clef.TrebleClef(),
                        "bass": m21_clef.BassClef(),
                        "alto": m21_clef.AltoClef()
                    }
                    if clef in clef_map:
                        first_part.insert(0, clef_map[clef])

                # Export to requested format
                if output_format == "musicxml":
                    score.write('musicxml', fp=output_path)

                elif output_format == "lilypond":
                    score.write('lilypond', fp=output_path)

                elif output_format == "pdf":
                    # PDF requires LilyPond to be installed
                    try:
                        score.write('lilypond.pdf', fp=output_path)
                    except Exception as e:
                        return {
                            "success": False,
                            "error": str(e),
                            "message": "PDF export requires LilyPond to be installed. Visit http://lilypond.org for installation instructions for your platform."
                        }

                elif output_format == "png":
                    # PNG also requires LilyPond
                    try:
                        score.write('lilypond.png', fp=output_path)
                    except Exception as e:
                        return {
                            "success": False,
                            "error": str(e),
                            "message": "PNG export requires LilyPond to be installed. Visit http://lilypond.org for installation instructions for your platform."
                        }

                else:
                    return {
                        "success": False,
                        "error": f"Unknown format: {output_format}",
                        "message": "Format must be one of: musicxml, lilypond, pdf, png"
                    }

                # Get measure count
                measure_count = 0
                for part in score.parts:
                    measures = part.getElementsByClass('Measure')
                    measure_count = max(measure_count, len(measures))

                # Fix ownership of output file and directory
                fix_ownership(output_path)
                if output_dir:
                    fix_ownership(output_dir)

                return {
                    "success": True,
                    "output_path": output_path,
                    "format": output_format,
                    "measure_count": measure_count,
                    "title": title,
                    "composer": composer,
                    "key_signature": key_signature,
                    "time_signature": time_signature,
                    "tempo": tempo,
                    "message": f"Successfully exported notation to {output_format} format"
                }

            except ImportError:
                return {
                    "success": False,
                    "error": "music21 not installed",
                    "message": "Please install music21: pip install music21"
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to export notation: {str(e)}"
            }
