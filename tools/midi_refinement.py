"""
MIDI refinement tool for cleaning up MIDI files.
"""

import os
from typing import Dict, Any, Optional, List

import numpy as np
from mido import MidiFile, MidiTrack, Message, MetaMessage
from fastmcp import FastMCP

from utils import fix_ownership


def register_tools(mcp: FastMCP):
    """Register MIDI refinement tools with the MCP server."""
    
    @mcp.tool()
    def refine_midi(
        midi_path: str,
        output_path: str = "./output/refined.mid",
        operations: Optional[List[str]] = None,
        quantize_grid: str = "16th",
        min_note_duration: float = 0.1,
        velocity_smoothing: int = 5,
        transpose: int = 0,
        tempo_scale: float = 1.0
    ) -> Dict[str, Any]:
        """
        Clean up and refine extracted MIDI files for better playback and notation.

        Args:
            midi_path: Path to the MIDI file to refine
            output_path: Path to save refined MIDI (default: ./output/refined.mid)
            operations: List of operations to apply (default: ["quantize", "remove_short_notes", "smooth_velocities"]).
                Available operations: "quantize", "remove_short_notes", "smooth_velocities"
            quantize_grid: Note duration for quantization - "32nd", "16th", "8th", "quarter" (default: 16th)
            min_note_duration: Minimum note length in seconds (default: 0.1)
            velocity_smoothing: Window size for velocity smoothing (default: 5)
            transpose: Semitones to transpose (+/-). Applied automatically when non-zero (default: 0)
            tempo_scale: Time stretch factor. Applied automatically when not 1.0 (default: 1.0)

        Returns:
            Dictionary with refined MIDI path, operations applied, and note counts
        """
        try:
            # Validate input file exists
            if not os.path.isfile(midi_path):
                return {
                    "success": False,
                    "error": f"MIDI file not found: {midi_path}",
                    "message": f"The specified MIDI file does not exist: {midi_path}"
                }

            # Set default operations
            if operations is None:
                operations = ["quantize", "remove_short_notes", "smooth_velocities"]

            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Load MIDI file
            midi_file = MidiFile(midi_path)

            # Get original note count
            original_note_count = sum(
                1 for track in midi_file.tracks
                for msg in track if msg.type == 'note_on' and msg.velocity > 0
            )

            # Get tempo and ticks per beat for calculations
            ticks_per_beat = midi_file.ticks_per_beat
            tempo_bpm = 120  # default

            # Find tempo in MIDI file
            tempo_found = False
            for track in midi_file.tracks:
                for msg in track:
                    if msg.type == 'set_tempo':
                        tempo_bpm = 60_000_000 / msg.tempo
                        tempo_found = True
                        break
                if tempo_found:
                    break

            # Calculate quantization grid in ticks
            grid_map = {
                "32nd": ticks_per_beat / 8,
                "16th": ticks_per_beat / 4,
                "8th": ticks_per_beat / 2,
                "quarter": ticks_per_beat
            }
            quantize_ticks = grid_map.get(quantize_grid, ticks_per_beat / 4)

            # Process each track
            for track in midi_file.tracks:
                notes = []
                current_time = 0

                # Extract notes with absolute timing
                for msg in track:
                    current_time += msg.time
                    if msg.type in ['note_on', 'note_off']:
                        notes.append({
                            'type': msg.type,
                            'note': msg.note,
                            'velocity': msg.velocity,
                            'time': current_time,
                            'original_msg': msg
                        })

                # Apply operations
                if "quantize" in operations:
                    # Quantize note timings
                    for note in notes:
                        quantized_time = round(note['time'] / quantize_ticks) * quantize_ticks
                        note['time'] = int(quantized_time)

                if transpose != 0:
                    # Transpose notes
                    for note in notes:
                        note['note'] = max(0, min(127, note['note'] + transpose))

                if tempo_scale != 1.0:
                    # Scale timing
                    for note in notes:
                        note['time'] = int(note['time'] * tempo_scale)

                if "smooth_velocities" in operations and velocity_smoothing > 1:
                    # Smooth velocities using moving average
                    velocities = [n['velocity'] for n in notes if n['type'] == 'note_on' and n['velocity'] > 0]
                    if velocities:
                        smoothed = []
                        for i in range(len(velocities)):
                            start = max(0, i - velocity_smoothing // 2)
                            end = min(len(velocities), i + velocity_smoothing // 2 + 1)
                            avg = int(np.mean(velocities[start:end]))
                            smoothed.append(avg)

                        # Apply smoothed velocities (only to note_on with velocity > 0)
                        note_on_idx = 0
                        for note in notes:
                            if note['type'] == 'note_on' and note['velocity'] > 0:
                                note['velocity'] = smoothed[note_on_idx]
                                note_on_idx += 1

                if "remove_short_notes" in operations:
                    # Remove notes shorter than min_note_duration
                    # Match note_on with note_off events
                    note_pairs = []
                    active_notes = {}

                    for note in notes:
                        if note['type'] == 'note_on' and note['velocity'] > 0:
                            active_notes[note['note']] = note
                        elif note['type'] == 'note_off' or (note['type'] == 'note_on' and note['velocity'] == 0):
                            if note['note'] in active_notes:
                                note_on = active_notes[note['note']]
                                duration_ticks = note['time'] - note_on['time']
                                duration_seconds = (duration_ticks / ticks_per_beat) * (60 / tempo_bpm)

                                if duration_seconds >= min_note_duration:
                                    note_pairs.append((note_on, note))

                                del active_notes[note['note']]

                    # Rebuild notes list from filtered pairs
                    notes = []
                    for note_on, note_off in note_pairs:
                        notes.append(note_on)
                        notes.append(note_off)

                    # Sort by time
                    notes.sort(key=lambda x: x['time'])

                # Preserve non-note messages (e.g., program changes, control changes, other meta)
                preserved_events = []
                abs_time = 0
                tempo_events_present = False
                for msg in track:
                    abs_time += msg.time
                    # Keep all messages except note events; track whether any tempo changes exist
                    if not (msg.type in ('note_on', 'note_off')):
                        preserved_events.append((abs_time, msg.copy()))
                        if getattr(msg, "is_meta", False) and msg.type == 'set_tempo':
                            tempo_events_present = True

                # Rebuild track with relative timing
                track.clear()

                # If no tempo events were present, add one based on tempo_bpm at time 0
                if not tempo_events_present:
                    track.append(MetaMessage('set_tempo', tempo=int(60_000_000 / tempo_bpm), time=0))

                # Merge preserved non-note events with refined note events using absolute time
                merged_events = []
                for note in notes:
                    merged_events.append((
                        note['time'],
                        'note',
                        note
                    ))
                for event_time, msg in preserved_events:
                    merged_events.append((
                        event_time,
                        'other',
                        msg
                    ))

                # Sort all events by absolute time
                merged_events.sort(key=lambda x: x[0])

                # Convert back to delta times and append to track
                last_time = 0
                for event_time, kind, data in merged_events:
                    delta_time = int(event_time - last_time)
                    if kind == 'note':
                        track.append(Message(
                            data['type'],
                            note=data['note'],
                            velocity=data['velocity'],
                            time=delta_time
                        ))
                    else:
                        msg = data.copy()
                        msg.time = delta_time
                        track.append(msg)
                    last_time = event_time

            # Save refined MIDI
            midi_file.save(output_path)

            # Get refined note count
            refined_note_count = sum(
                1 for track in midi_file.tracks
                for msg in track if msg.type == 'note_on' and msg.velocity > 0
            )

            # Fix ownership of output file and directory
            fix_ownership(output_path)
            if output_dir:
                fix_ownership(output_dir)

            return {
                "success": True,
                "output_path": output_path,
                "operations_applied": operations,
                "note_count_before": original_note_count,
                "note_count_after": refined_note_count,
                "quantize_grid": quantize_grid if "quantize" in operations else None,
                "transpose_semitones": transpose if "transpose" in operations else None,
                "message": f"Successfully refined MIDI with {len(operations)} operations"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to refine MIDI: {str(e)}"
            }
