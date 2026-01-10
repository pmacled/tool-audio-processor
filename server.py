#!/usr/bin/env python3
"""
MCP Server for Audio Layer Manipulation
Provides tools for separating, analyzing, synthesizing, and manipulating audio layers.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from fastmcp import FastMCP
import torch
import torchaudio
import librosa
import numpy as np
import soundfile as sf
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import save_audio
import mido
from mido import MidiFile, MidiTrack, Message
import pretty_midi

# Initialize FastMCP server
mcp = FastMCP("audio-processor")

# Global model cache
_demucs_model = None
_device = None


def get_device():
    """Get the appropriate device (CUDA if available, otherwise CPU)."""
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def get_demucs_model():
    """Load and cache the Demucs model."""
    global _demucs_model
    if _demucs_model is None:
        device = get_device()
        _demucs_model = get_model(name='htdemucs')
        _demucs_model.to(device)
        _demucs_model.eval()
    return _demucs_model


@mcp.tool()
def separate_audio_layers(
    audio_path: str,
    output_dir: str = "/app/output",
    model: str = "htdemucs"
) -> Dict[str, Any]:
    """
    Separate audio into vocals, drums, bass, and other layers using Demucs.
    
    Args:
        audio_path: Path to the input audio file
        output_dir: Directory to save separated layers (default: /app/output)
        model: Demucs model to use (default: htdemucs)
    
    Returns:
        Dictionary with paths to separated layer files and metadata
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        device = get_device()
        
        # Load model
        demucs_model = get_demucs_model()
        
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
    output_path: str = "/app/output/synthesized.wav",
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
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        # Ensure output directory exists (if a directory is specified)
        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        
        # Save audio
        sf.write(output_path, audio, sample_rate)
        
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
    output_path: str = "/app/output/replaced_mix.wav"
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
        
        # Load all layers
        layer_audio = {}
        sr = None
        max_length = 0
        
        for name, path in layers.items():
            if name == layer_to_replace:
                # Load the new layer instead
                audio, sr = librosa.load(new_layer_path, sr=None)
            else:
                audio, sr = librosa.load(path, sr=sr)
            
            layer_audio[name] = audio
            max_length = max(max_length, len(audio))
        
        # Pad all layers to the same length
        for name in layer_audio:
            if len(layer_audio[name]) < max_length:
                layer_audio[name] = np.pad(
                    layer_audio[name], 
                    (0, max_length - len(layer_audio[name]))
                )
        
        # Mix all layers
        mixed = np.zeros(max_length)
        for audio in layer_audio.values():
            mixed += audio
        
        # Normalize
        mixed = mixed / len(layer_audio)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save output
        sf.write(output_path, mixed, sr)
        
        return {
            "success": True,
            "output_path": output_path,
            "replaced_layer": layer_to_replace,
            "sample_rate": sr,
            "duration": float(max_length / sr),
            "message": f"Successfully replaced {layer_to_replace} layer"
        }
        
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
    output_path: str = "/app/output/modified.wav",
    **effect_params
) -> Dict[str, Any]:
    """
    Apply audio effects to a layer.
    
    Args:
        audio_path: Path to the audio file to modify
        effect: Effect to apply - "pitch_shift", "time_stretch", "reverb", "normalize", "fade"
        output_path: Path to save modified audio
        effect_params: Effect-specific parameters:
            - pitch_shift: steps (int, semitones)
            - time_stretch: rate (float, speed multiplier)
            - fade: fade_in (float, seconds), fade_out (float, seconds)
            - normalize: target_db (float, dB level)
    
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
            steps = effect_params.get("steps", 0)
            modified = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
            applied_params["steps"] = steps
            
        elif effect == "time_stretch":
            rate = effect_params.get("rate", 1.0)
            modified = librosa.effects.time_stretch(y, rate=rate)
            applied_params["rate"] = rate
            
        elif effect == "normalize":
            target_db = effect_params.get("target_db", -20.0)
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
            fade_in = effect_params.get("fade_in", 0.0)
            fade_out = effect_params.get("fade_out", 0.0)
            
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
            decay = effect_params.get("decay", 0.5)
            delay_samples = int(0.05 * sr)  # 50ms delay
            
            reverb_response = np.zeros(delay_samples)
            reverb_response[0] = 1.0
            reverb_response[-1] = decay
            
            modified = np.convolve(modified, reverb_response, mode='same')
            applied_params["decay"] = decay
        else:
            return {
                "success": False,
                "error": f"Unknown effect: {effect}",
                "message": "Effect must be one of: pitch_shift, time_stretch, reverb, normalize, fade"
            }
        
        # Clip to prevent distortion
        modified = np.clip(modified, -1.0, 1.0)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save output
        sf.write(output_path, modified, sr)
        
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
    output_path: str = "/app/output/mixed.wav",
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
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save output
        sf.write(output_path, mixed, sr)
        
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


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
