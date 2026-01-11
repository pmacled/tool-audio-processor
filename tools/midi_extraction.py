"""
MIDI extraction tool for converting audio to MIDI.
"""

import os
from typing import Dict, Any, Optional

import librosa
import numpy as np
from fastmcp import FastMCP

from utils import fix_ownership


def _create_midi_from_notes(midi_notes, output_midi_path, tempo, min_note_duration):
    """
    Helper function to create a MIDI file from a list of note tuples.
    
    Args:
        midi_notes: List of (time, note) or (time, note, confidence) tuples
        output_midi_path: Path to save the MIDI file
        tempo: Tempo in BPM
        min_note_duration: Minimum note duration in seconds for filtering
    
    Returns:
        Tuple of (note_count, pitch_min, pitch_max)
    """
    from mido import MidiFile, MidiTrack, Message, MetaMessage
    
    # Create MIDI file
    midi_file = MidiFile()
    track = MidiTrack()
    midi_file.tracks.append(track)
    
    # Set tempo
    microseconds_per_beat = int(60_000_000 / tempo)
    track.append(MetaMessage('set_tempo', tempo=microseconds_per_beat, time=0))
    
    # Add notes
    if midi_notes:
        # Normalize midi_notes to (time, note) tuples if they have confidence
        normalized_notes = []
        for note_data in midi_notes:
            if len(note_data) == 3:
                # (time, note, confidence) - extract time and note
                normalized_notes.append((note_data[0], note_data[1]))
            else:
                # Already (time, note)
                normalized_notes.append(note_data)
        
        ticks_per_second = midi_file.ticks_per_beat * tempo / 60
        last_event_time = 0
        
        # Process notes by grouping consecutive same-pitch notes
        i = 0
        while i < len(normalized_notes):
            current_start, current_note = normalized_notes[i]
            
            # Find where this note ends (when pitch changes or list ends)
            j = i + 1
            while j < len(normalized_notes) and normalized_notes[j][1] == current_note:
                j += 1
            
            # Calculate note duration
            if j < len(normalized_notes):
                note_duration = normalized_notes[j][0] - current_start
            else:
                # Last note - use min_note_duration
                note_duration = min_note_duration
            
            # Only add note if it meets minimum duration
            if note_duration >= min_note_duration:
                # Add note on
                abs_start_time = int(current_start * ticks_per_second)
                delta_time = abs_start_time - last_event_time
                track.append(Message('note_on', note=current_note, velocity=64, time=delta_time))
                last_event_time = abs_start_time
                
                # Add note off
                abs_end_time = int((current_start + note_duration) * ticks_per_second)
                delta_time = abs_end_time - last_event_time
                track.append(Message('note_off', note=current_note, velocity=64, time=delta_time))
                last_event_time = abs_end_time
            
            # Move to next different note
            i = j
        
        # Save MIDI file
        midi_file.save(output_midi_path)
        
        # Get statistics
        note_count = len([msg for msg in track if msg.type == 'note_on' and msg.velocity > 0])
        all_pitches = [msg.note for msg in track if msg.type == 'note_on' and msg.velocity > 0]
        pitch_min = min(all_pitches) if all_pitches else 0
        pitch_max = max(all_pitches) if all_pitches else 0
        
        return note_count, pitch_min, pitch_max
    else:
        return 0, 0, 0


def register_tools(mcp: FastMCP):
    """Register MIDI extraction tools with the MCP server."""
    
    @mcp.tool()
    def extract_melody_to_midi(
        audio_path: str,
        output_midi_path: str = "./output/melody.mid",
        method: str = "basic-pitch",
        voicing_threshold: float = 0.5,
        tempo: Optional[float] = None,
        min_note_duration: float = 0.1
    ) -> Dict[str, Any]:
        """
        Extract melody from monophonic or dominant melody audio to MIDI.
        This is the critical tool for converting vocals or instruments to MIDI notation.

        Args:
            audio_path: Path to the audio file to convert
            output_midi_path: Path to save the MIDI file (default: ./output/melody.mid)
            method: Pitch detection method - "basic-pitch", "crepe", "pyin" (default: basic-pitch)
            voicing_threshold: Confidence threshold for pitch detection (0-1, default: 0.5)
            tempo: BPM for quantization (if None, will be detected, default: None)
            min_note_duration: Minimum note length in seconds (default: 0.1)

        Returns:
            Dictionary with MIDI path, note count, duration, pitch range, and metadata
        """
        try:
            # Validate input file exists
            if not os.path.isfile(audio_path):
                return {
                    "success": False,
                    "error": f"Audio file not found: {audio_path}",
                    "message": f"The specified audio file does not exist: {audio_path}"
                }

            # Ensure output directory exists
            output_dir = os.path.dirname(output_midi_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)

            # Detect tempo if not provided
            if tempo is None:
                detected_tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                tempo = float(detected_tempo)

            # Method selection and pitch extraction
            if method == "basic-pitch":
                # Use basic-pitch for audio-to-MIDI conversion
                try:
                    from basic_pitch.inference import predict
                    from basic_pitch import ICASSP_2022_MODEL_PATH

                    # Predict MIDI from audio
                    model_output, midi_data, note_events = predict(
                        audio_path,
                        ICASSP_2022_MODEL_PATH
                    )

                    # Save MIDI file
                    midi_data.write(output_midi_path)

                    # Get note statistics
                    note_count = sum(len(inst.notes) for inst in midi_data.instruments)

                    # Get pitch range
                    all_pitches = [note.pitch for inst in midi_data.instruments for note in inst.notes]
                    if all_pitches:
                        pitch_min = min(all_pitches)
                        pitch_max = max(all_pitches)
                    else:
                        pitch_min = pitch_max = 0

                except ImportError:
                    return {
                        "success": False,
                        "error": "basic-pitch not installed",
                        "message": "Please install basic-pitch: pip install basic-pitch"
                    }

            elif method == "crepe":
                # Use CREPE for pitch tracking
                try:
                    import crepe

                    # Predict pitch
                    time, frequency, confidence, activation = crepe.predict(
                        y, sr, viterbi=True
                    )

                    # Filter by confidence threshold
                    voiced_mask = confidence > voicing_threshold

                    # Convert Hz to MIDI note numbers
                    midi_notes = []
                    for i, (f, conf, voiced) in enumerate(zip(frequency, confidence, voiced_mask)):
                        if voiced and f > 0:
                            midi_note = int(round(librosa.hz_to_midi(f)))
                            midi_notes.append((time[i], midi_note, conf))

                    # Create MIDI file using helper function
                    if midi_notes:
                        note_count, pitch_min, pitch_max = _create_midi_from_notes(
                            midi_notes, output_midi_path, tempo, min_note_duration
                        )
                    else:
                        return {
                            "success": False,
                            "error": "No notes detected",
                            "message": "No pitched content found in audio. Try lowering voicing_threshold."
                        }

                except ImportError:
                    return {
                        "success": False,
                        "error": "crepe not installed",
                        "message": "Please install crepe: pip install crepe"
                    }

            elif method == "pyin":
                # Use librosa's pYIN algorithm
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    y,
                    fmin=librosa.note_to_hz('C2'),
                    fmax=librosa.note_to_hz('C7'),
                    sr=sr
                )

                # Filter by voicing probability
                voiced_mask = voiced_probs > voicing_threshold

                # Convert Hz to MIDI notes
                midi_notes = []
                hop_length = 512  # default for librosa.pyin
                times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)

                for i, (f, voiced) in enumerate(zip(f0, voiced_mask)):
                    if voiced and not np.isnan(f) and f > 0:
                        midi_note = int(round(librosa.hz_to_midi(f)))
                        midi_notes.append((times[i], midi_note))

                # Create MIDI file using helper function
                if midi_notes:
                    note_count, pitch_min, pitch_max = _create_midi_from_notes(
                        midi_notes, output_midi_path, tempo, min_note_duration
                    )
                else:
                    return {
                        "success": False,
                        "error": "No notes detected",
                        "message": "No pitched content found in audio. Try lowering voicing_threshold."
                    }

            else:
                return {
                    "success": False,
                    "error": f"Unknown method: {method}",
                    "message": "Method must be one of: basic-pitch, crepe, pyin"
                }

            # Fix ownership of output file and directory
            fix_ownership(output_midi_path)
            if output_dir:
                fix_ownership(output_dir)

            return {
                "success": True,
                "midi_path": output_midi_path,
                "note_count": note_count,
                "duration": float(duration),
                "pitch_range": {"min": int(pitch_min), "max": int(pitch_max)},
                "method_used": method,
                "tempo_bpm": float(tempo),
                "message": f"Successfully extracted melody to MIDI using {method} method"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to extract melody to MIDI: {str(e)}"
            }
