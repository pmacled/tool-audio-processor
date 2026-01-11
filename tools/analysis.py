"""
Audio analysis tool using librosa.
"""

import os
from typing import Dict, Any

import librosa
import numpy as np
from fastmcp import FastMCP


def register_tools(mcp: FastMCP):
    """Register analysis tools with the MCP server."""
    
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
