#!/usr/bin/env python3
"""
MCP Server for Audio Layer Manipulation
Provides tools for separating, analyzing, synthesizing, and manipulating audio layers.
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastmcp import FastMCP
import torch
import torchaudio
import librosa
import numpy as np
import soundfile as sf
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import save_audio
import pretty_midi
from mido import MidiFile, MidiTrack, Message, MetaMessage

# Initialize FastMCP server
mcp = FastMCP("audio-processor")

# Global model cache
_demucs_model = None
_device = None


def get_workspace_owner():
    """Get the UID/GID from environment variables (HOST_UID and HOST_GID)."""
    try:
        uid = int(os.environ.get('HOST_UID', 0))
        gid = int(os.environ.get('HOST_GID', 0))
        # Only return valid non-root UID/GID
        if uid > 0 and gid > 0:
            return uid, gid
        return None, None
    except (TypeError, ValueError) as e:
        print(
            f"Warning: Invalid HOST_UID/HOST_GID environment values; "
            f"skipping ownership changes: {e}",
            flush=True,
        )
        return None, None


def fix_ownership(path):
    """Change ownership of path to match /workspace owner."""
    uid, gid = get_workspace_owner()
    if uid is not None and gid is not None:
        try:
            # If path is a directory, chown recursively
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    os.chown(root, uid, gid)
                    for f in files:
                        os.chown(os.path.join(root, f), uid, gid)
            else:
                os.chown(path, uid, gid)
        except Exception as e:
            # Log but don't fail - this is a best-effort operation
            print(f"Warning: Could not change ownership of {path}: {e}", flush=True)


def get_device():
    """Get the appropriate device (CUDA if available, otherwise CPU)."""
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def get_demucs_model(model_name: str = 'htdemucs'):
    """Load and cache the Demucs model."""
    global _demucs_model
    if _demucs_model is None:
        device = get_device()
        _demucs_model = get_model(name=model_name)
        _demucs_model.to(device)
        _demucs_model.eval()
    return _demucs_model


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


@mcp.tool()
def analyze_layer(
    audio_path: str,
    analysis_type: str = "all"
) -> Dict[str, Any]:
    """
    Analyze an audio layer to extract musical features like notes, tempo, rhythm, and key.
    
    Args:
        audio_path: Path to the audio file to analyze
        analysis_type: Type of analysis - "all", "tempo", "key", "notes", "rhythm" (default: all)
    
    Returns:
        Dictionary with analysis results including tempo, key, notes, and rhythm information
    """
    try:
        # Validate input file exists
        if not os.path.isfile(audio_path):
            return {
                "success": False,
                "error": f"Audio file not found: {audio_path}",
                "message": f"The specified audio file does not exist: {audio_path}"
            }
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        results = {
            "success": True,
            "file": audio_path,
            "sample_rate": sr,
            "duration": float(librosa.get_duration(y=y, sr=sr))
        }
        
        # Tempo and beat analysis
        if analysis_type in ["all", "tempo", "rhythm"]:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            results["tempo"] = float(tempo)
            results["beats"] = beats.tolist() if analysis_type == "all" else len(beats)
        
        # Key detection (using chroma features)
        if analysis_type in ["all", "key"]:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key_idx = np.argmax(np.sum(chroma, axis=1))
            results["key"] = key_names[key_idx]
        
        # Note/pitch analysis
        if analysis_type in ["all", "notes"]:
            # Extract pitch using piptrack
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            
            # Get the most prominent pitches
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(float(pitch))
            
            if pitch_values:
                results["pitch_mean"] = float(np.mean(pitch_values))
                results["pitch_std"] = float(np.std(pitch_values))
                results["pitch_min"] = float(np.min(pitch_values))
                results["pitch_max"] = float(np.max(pitch_values))
        
        # Rhythm/onset analysis
        if analysis_type in ["all", "rhythm"]:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            results["onsets_count"] = len(onsets)
            results["onset_rate"] = float(len(onsets) / results["duration"])
        
        # Spectral features
        if analysis_type == "all":
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            results["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            results["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))
            
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            results["zero_crossing_rate_mean"] = float(np.mean(zero_crossing_rate))
        
        results["message"] = f"Successfully analyzed audio with type: {analysis_type}"
        return results
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to analyze audio: {str(e)}"
        }


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
        # First, separate the original mix
        temp_dir = tempfile.mkdtemp()
        
        try:
            separation_result = separate_audio_layers(original_mix_path, temp_dir)
            
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
            # Normalize reverb output to avoid harsh clipping from convolution overs
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

                # Create MIDI file using mido
                midi_file = MidiFile()
                track = MidiTrack()
                midi_file.tracks.append(track)

                # Set tempo
                microseconds_per_beat = int(60_000_000 / tempo)
                track.append(MetaMessage('set_tempo', tempo=microseconds_per_beat, time=0))

                # Add notes (simplified: just note on/off events)
                if midi_notes:
                    current_note = None
                    current_start = 0
                    ticks_per_second = midi_file.ticks_per_beat * tempo / 60
                    last_event_time = 0

                    for i, (t, note, conf) in enumerate(midi_notes):
                        if current_note is None:
                            # Start new note
                            current_note = note
                            current_start = t
                        elif note != current_note or i == len(midi_notes) - 1:
                            # End previous note
                            note_duration = t - current_start

                            # Filter short notes
                            if note_duration >= min_note_duration:
                                # Calculate delta time from last event
                                abs_start_time = int(current_start * ticks_per_second)
                                delta_time = abs_start_time - last_event_time
                                track.append(Message('note_on', note=current_note, velocity=64, time=delta_time))
                                last_event_time = abs_start_time

                                abs_end_time = int((current_start + note_duration) * ticks_per_second)
                                delta_time = abs_end_time - last_event_time
                                track.append(Message('note_off', note=current_note, velocity=64, time=delta_time))
                                last_event_time = abs_end_time

                            # Start new note (unless this is the last iteration and we just ended the final note)
                            if i < len(midi_notes) - 1:
                                current_note = note
                                current_start = t
                            else:
                                # Handle the very last note
                                if note != current_note:
                                    current_note = note
                                    current_start = t
                                    # Add the final note
                                    abs_start_time = int(current_start * ticks_per_second)
                                    delta_time = abs_start_time - last_event_time
                                    track.append(Message('note_on', note=current_note, velocity=64, time=delta_time))
                                    last_event_time = abs_start_time
                                    
                                    # Estimate end time (use min_note_duration)
                                    abs_end_time = int((current_start + min_note_duration) * ticks_per_second)
                                    delta_time = abs_end_time - last_event_time
                                    track.append(Message('note_off', note=current_note, velocity=64, time=delta_time))
                                    last_event_time = abs_end_time

                    # Save MIDI file
                    midi_file.save(output_midi_path)

                    # Get statistics
                    note_count = len([msg for msg in track if msg.type == 'note_on' and msg.velocity > 0])
                    all_pitches = [msg.note for msg in track if msg.type == 'note_on' and msg.velocity > 0]
                    pitch_min = min(all_pitches) if all_pitches else 0
                    pitch_max = max(all_pitches) if all_pitches else 0
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

            # Create MIDI file
            midi_file = MidiFile()
            track = MidiTrack()
            midi_file.tracks.append(track)

            # Set tempo
            microseconds_per_beat = int(60_000_000 / tempo)
            track.append(MetaMessage('set_tempo', tempo=microseconds_per_beat, time=0))

            # Add notes
            if midi_notes:
                current_note = None
                current_start = 0
                ticks_per_second = midi_file.ticks_per_beat * tempo / 60
                last_event_time = 0

                for i, (t, note) in enumerate(midi_notes):
                    if current_note is None:
                        current_note = note
                        current_start = t
                    elif note != current_note or i == len(midi_notes) - 1:
                        note_duration = t - current_start

                        if note_duration >= min_note_duration:
                            # Calculate delta time from last event
                            abs_start_time = int(current_start * ticks_per_second)
                            delta_time = abs_start_time - last_event_time
                            track.append(Message('note_on', note=current_note, velocity=64, time=delta_time))
                            last_event_time = abs_start_time

                            abs_end_time = int((current_start + note_duration) * ticks_per_second)
                            delta_time = abs_end_time - last_event_time
                            track.append(Message('note_off', note=current_note, velocity=64, time=delta_time))
                            last_event_time = abs_end_time

                        # Start new note (unless this is the last iteration and we just ended the final note)
                        if i < len(midi_notes) - 1:
                            current_note = note
                            current_start = t
                        else:
                            # Handle the very last note
                            if note != current_note:
                                current_note = note
                                current_start = t
                                # Add the final note
                                abs_start_time = int(current_start * ticks_per_second)
                                delta_time = abs_start_time - last_event_time
                                track.append(Message('note_on', note=current_note, velocity=64, time=delta_time))
                                last_event_time = abs_start_time
                                
                                # Estimate end time (use min_note_duration)
                                abs_end_time = int((current_start + min_note_duration) * ticks_per_second)
                                delta_time = abs_end_time - last_event_time
                                track.append(Message('note_off', note=current_note, velocity=64, time=delta_time))
                                last_event_time = abs_end_time

                midi_file.save(output_midi_path)

                # Get statistics
                note_count = len([msg for msg in track if msg.type == 'note_on' and msg.velocity > 0])
                all_pitches = [msg.note for msg in track if msg.type == 'note_on' and msg.velocity > 0]
                pitch_min = min(all_pitches) if all_pitches else 0
                pitch_max = max(all_pitches) if all_pitches else 0
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
        operations: List of operations to apply (default: ["quantize", "remove_short_notes", "smooth_velocities"])
        quantize_grid: Note duration for quantization - "32nd", "16th", "8th", "quarter" (default: 16th)
        min_note_duration: Minimum note length in seconds (default: 0.1)
        velocity_smoothing: Window size for velocity smoothing (default: 5)
        transpose: Semitones to transpose (+/-) (default: 0)
        tempo_scale: Time stretch factor (default: 1.0)

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

            if "transpose" in operations and transpose != 0:
                # Transpose notes
                for note in notes:
                    note['note'] = max(0, min(127, note['note'] + transpose))

            if "tempo_scale" in operations and tempo_scale != 1.0:
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
            for msg in track:
                abs_time += msg.time
                # Keep all messages except note events and set_tempo (which we explicitly re-add)
                if not (
                    msg.type in ('note_on', 'note_off') or
                    (getattr(msg, "is_meta", False) and msg.type == 'set_tempo')
                ):
                    preserved_events.append((abs_time, msg.copy()))

            # Rebuild track with relative timing
            track.clear()

            # Re-add tempo message based on tempo_bpm at time 0
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

                # Add tempo marking
                if tempo:
                    tempo_mark = m21_tempo.MetronomeMark(number=tempo)
                    first_part.insert(0, tempo_mark)

                # Set clef
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
                        "message": "PDF export requires LilyPond to be installed. Install with: sudo apt-get install lilypond"
                    }

            elif output_format == "png":
                # PNG also requires LilyPond
                try:
                    score.write('lilypond.png', fp=output_path)
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "message": "PNG export requires LilyPond to be installed. Install with: sudo apt-get install lilypond"
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


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
