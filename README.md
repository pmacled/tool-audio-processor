# Audio Processor MCP Server

A Docker-based Model Context Protocol (MCP) server for advanced audio layer manipulation, powered by AI and deep learning.

## Features

This MCP server provides nine powerful tools for audio processing:

1. **separate_audio_layers** - Split audio into vocals, drums, bass, and other layers using Demucs
2. **analyze_layer** - Extract musical features including notes, tempo, rhythm, and key using librosa
3. **extract_melody_to_midi** - Convert audio to MIDI notation using pitch detection
4. **refine_midi** - Clean up and refine MIDI files with quantization, smoothing, and transposition
5. **export_notation** - Convert MIDI to human-readable music notation (MusicXML, LilyPond, PDF, PNG)
6. **synthesize_instrument_layer** - Generate new instrument audio from MIDI files
7. **replace_layer** - Swap out specific audio layers in a mix
8. **modify_layer** - Apply effects like pitch shift, time stretch, reverb, and normalization
9. **mix_layers** - Combine multiple audio layers with volume control

## Technology Stack

- **Base Image**: NVIDIA CUDA 12.1.0 with cuDNN 8 on Ubuntu 22.04
- **Python**: 3.11+
- **Deep Learning**: PyTorch, Torchaudio
- **Audio Processing**: Demucs (vocal separation), Librosa (analysis)
- **MCP Framework**: FastMCP
- **MIDI Processing**: Pretty MIDI, Mido, Basic Pitch, CREPE, pyFluidSynth
- **Music Notation**: Music21, LilyPond (for PDF/PNG export)

## Container Architecture

The container uses a **dual-workspace design** to prevent conflicts:
- **`/app`**: Container's internal workspace containing the MCP server and application code
- **`/workspace`**: User's workspace mounted from the local directory for audio files and processing

This separation ensures that mounting your local directory doesn't interfere with the container's built-in files.

## Prerequisites

- Docker with NVIDIA GPU support (nvidia-docker2)
- NVIDIA GPU with CUDA 12.1 support (for accelerated processing)
- For CPU-only usage, the image will work but processing will be slower

## Building the Docker Image

```bash
docker build -t audio-processor-mcp .
```

## Running the Server

### Using Docker Compose (Recommended)

```bash
# Set your user ID and group ID, then run docker-compose
export UID=$(id -u)
export GID=$(id -g)
docker-compose up
```

### Using Docker Directly

#### With GPU Support

```bash
docker run --gpus all -i -e HOST_UID=$(id -u) -e HOST_GID=$(id -g) -v $(pwd):/workspace -w /workspace audio-processor-mcp
```

#### CPU Only

```bash
docker run -i -e HOST_UID=$(id -u) -e HOST_GID=$(id -g) -v $(pwd):/workspace -w /workspace audio-processor-mcp
```

**Note:** The container runs as root internally but automatically adjusts file ownership to match your user (via HOST_UID and HOST_GID environment variables), ensuring output files are owned by your user account. The entire working directory is mounted, allowing the container to create input/output directories as needed.

### MCP Client Configuration

Add this to your MCP client configuration (e.g., Claude Code). Choose either the GPU or CPU version based on your system:

#### Option 1: With GPU Support (Recommended for faster processing)

```json
{
  "mcpServers": {
    "audio-processor": {
      "command": "docker",
      "args": [
        "run",
        "--rm",
        "-i",
        "-e",
        "HOST_UID=1000",
        "-e",
        "HOST_GID=1000",
        "--gpus",
        "all",
        "-v",
        "${PWD}:/workspace",
        "-w",
        "/workspace",
        "ghcr.io/pmacled/tool-audio-processor:latest"
      ],
      "description": "MCP server for audio layer manipulation with CUDA acceleration"
    }
  }
}
```

**Requirements**: NVIDIA GPU with nvidia-docker2 installed

**Note**: Update HOST_UID and HOST_GID to match your user (run `id -u` and `id -g` to find your values).

#### Option 2: CPU Only

```json
{
  "mcpServers": {
    "audio-processor": {
      "command": "docker",
      "args": [
        "run",
        "--rm",
        "-i",
        "-e",
        "HOST_UID=1000",
        "-e",
        "HOST_GID=1000",
        "-v",
        "${PWD}:/workspace",
        "-w",
        "/workspace",
        "ghcr.io/pmacled/tool-audio-processor:latest"
      ],
      "description": "MCP server for audio layer manipulation (CPU-only)"
    }
  }
}
```

**Notes**:
- Processing will be slower without GPU acceleration, but works on any system with Docker.
- The container runs as root internally but automatically adjusts file ownership to match your user (via HOST_UID and HOST_GID).
- Update HOST_UID and HOST_GID to match your user (run `id -u` and `id -g` to find your values).
- The entire working directory is mounted at `/workspace`, allowing you to work with audio files in your project structure without pre-creating directories.

See `mcp-config.json` for both configurations.

## MCP Tools Reference

### 1. separate_audio_layers

Separates audio into individual stems using the Demucs model.

**Parameters:**
- `audio_path` (str): Path to input audio file
- `output_dir` (str, optional): Output directory (default: ./output)
- `model` (str, optional): Demucs model to use (default: mdx)

**Returns:**
```json
{
  "success": true,
  "layers": {
    "vocals": "./output/song_vocals.wav",
    "drums": "./output/song_drums.wav",
    "bass": "./output/song_bass.wav",
    "other": "./output/song_other.wav"
  },
  "sample_rate": 44100,
  "device": "cuda:0"
}
```

### 2. analyze_layer

Analyzes an audio file to extract musical features.

**Parameters:**
- `audio_path` (str): Path to audio file
- `analysis_type` (str, optional): Type of analysis - "all", "tempo", "key", "notes", "rhythm" (default: all)

**Returns:**
```json
{
  "success": true,
  "tempo": 120.0,
  "key": "C",
  "pitch_mean": 440.0,
  "spectral_centroid_mean": 2000.5,
  "duration": 180.5
}
```

### 3. extract_melody_to_midi

Extracts melody from audio and converts it to MIDI notation.

**Parameters:**
- `audio_path` (str): Path to audio file to convert
- `output_midi_path` (str, optional): Output MIDI path (default: ./output/melody.mid)
- `method` (str, optional): Pitch detection method - "basic-pitch", "crepe", "pyin" (default: basic-pitch)
- `voicing_threshold` (float, optional): Confidence threshold for pitch detection 0-1 (default: 0.5)
- `tempo` (float, optional): BPM for quantization, auto-detected if None (default: None)
- `min_note_duration` (float, optional): Minimum note length in seconds (default: 0.1)

**Returns:**
```json
{
  "success": true,
  "midi_path": "./output/melody.mid",
  "note_count": 450,
  "duration": 120.5,
  "pitch_range": {"min": 60, "max": 84},
  "method_used": "basic-pitch",
  "tempo_bpm": 120.0
}
```

### 4. refine_midi

Cleans up and refines extracted MIDI files for better playback and notation.

**Parameters:**
- `midi_path` (str): Path to MIDI file to refine
- `output_path` (str, optional): Output path (default: ./output/refined.mid)
- `operations` (List[str], optional): Operations to apply (default: ["quantize", "remove_short_notes", "smooth_velocities"])
- `quantize_grid` (str, optional): Quantization grid - "32nd", "16th", "8th", "quarter" (default: 16th)
- `min_note_duration` (float, optional): Minimum note length in seconds (default: 0.1)
- `velocity_smoothing` (int, optional): Window size for velocity smoothing (default: 5)
- `transpose` (int, optional): Semitones to transpose +/- (default: 0)
- `tempo_scale` (float, optional): Time stretch factor (default: 1.0)

**Returns:**
```json
{
  "success": true,
  "output_path": "./output/refined.mid",
  "operations_applied": ["quantize", "remove_short_notes", "smooth_velocities"],
  "note_count_before": 450,
  "note_count_after": 380,
  "quantize_grid": "16th",
  "transpose_semitones": 2,
  "tempo_scale": 1.0
}
```

### 5. export_notation

Converts MIDI files to human-readable music notation formats.

**Parameters:**
- `midi_path` (str): Path to MIDI file to convert
- `output_path` (str, optional): Output file path (default: ./output/notation.musicxml)
- `output_format` (str, optional): Output format - "musicxml", "lilypond", "pdf", "png" (default: musicxml)
- `title` (str, optional): Title of the piece (default: None)
- `composer` (str, optional): Composer name (default: None)
- `key_signature` (str, optional): Key signature, e.g., "C", "D#", "Bb" (default: None)
- `time_signature` (str, optional): Time signature (default: 4/4)
- `tempo` (float, optional): Tempo in BPM for display (default: None)
- `clef` (str, optional): Clef type - "treble", "bass", "alto" (default: treble)

**Returns:**
```json
{
  "success": true,
  "output_path": "./output/notation.pdf",
  "format": "pdf",
  "measure_count": 32,
  "title": "My Melody",
  "composer": "John Doe",
  "key_signature": "C",
  "time_signature": "4/4",
  "tempo": 120.0
}
```

### 6. synthesize_instrument_layer

Generates audio from MIDI files.

**Parameters:**
- `midi_path` (str): Path to MIDI file
- `instrument` (str, optional): Instrument type (default: piano)
- `output_path` (str, optional): Output path (default: ./output/synthesized.wav)
- `sample_rate` (int, optional): Sample rate (default: 44100)

**Returns:**
```json
{
  "success": true,
  "output_path": "./output/synthesized.wav",
  "duration": 120.5,
  "total_notes": 1500
}
```

### 7. replace_layer

Replaces a specific layer in a mixed audio file.

**Parameters:**
- `original_mix_path` (str): Path to original mixed audio
- `layer_to_replace` (str): Layer to replace (vocals, drums, bass, or other)
- `new_layer_path` (str): Path to new layer audio
- `output_path` (str, optional): Output path (default: ./output/replaced_mix.wav)

**Returns:**
```json
{
  "success": true,
  "output_path": "./output/replaced_mix.wav",
  "replaced_layer": "vocals",
  "duration": 180.5
}
```

### 8. modify_layer

Applies audio effects to a layer.

**Parameters:**
- `audio_path` (str): Path to audio file
- `effect` (str): Effect type - "pitch_shift", "time_stretch", "reverb", "normalize", "fade"
- `output_path` (str, optional): Output path (default: ./output/modified.wav)
- `effect_params`: Effect-specific parameters
  - pitch_shift: `steps` (int) - semitones
  - time_stretch: `rate` (float) - speed multiplier
  - fade: `fade_in` (float), `fade_out` (float) - seconds
  - normalize: `target_db` (float) - dB level
  - reverb: `decay` (float) - decay amount

**Returns:**
```json
{
  "success": true,
  "output_path": "./output/modified.wav",
  "effect": "pitch_shift",
  "effect_params": {"steps": 2}
}
```

### 9. mix_layers

Combines multiple audio layers into a single mix.

**Parameters:**
- `layer_paths` (List[str]): List of audio file paths
- `output_path` (str, optional): Output path (default: ./output/mixed.wav)
- `layer_volumes` (List[float], optional): Volume multipliers for each layer
- `normalize_output` (bool, optional): Normalize output (default: true)

**Returns:**
```json
{
  "success": true,
  "output_path": "./output/mixed.wav",
  "num_layers": 4,
  "duration": 180.5
}
```

## Usage Examples

### Example 1: Separate and Analyze

```python
# Separate audio into layers
result = separate_audio_layers(
    audio_path="./input/song.wav",
    output_dir="./output"
)

# Analyze the vocal layer
analysis = analyze_layer(
    audio_path=result["layers"]["vocals"],
    analysis_type="all"
)
```

### Example 2: Replace Vocals

```python
# Replace vocals with a new recording
result = replace_layer(
    original_mix_path="./input/original.wav",
    layer_to_replace="vocals",
    new_layer_path="./input/new_vocals.wav",
    output_path="./output/new_mix.wav"
)
```

### Example 3: Apply Effects and Mix

```python
# Apply pitch shift to vocals
modified = modify_layer(
    audio_path="./output/vocals.wav",
    effect="pitch_shift",
    steps=2,
    output_path="./output/vocals_shifted.wav"
)

# Mix all layers with custom volumes
final_mix = mix_layers(
    layer_paths=[
        "./output/vocals_shifted.wav",
        "./output/drums.wav",
        "./output/bass.wav",
        "./output/other.wav"
    ],
    layer_volumes=[1.2, 1.0, 0.8, 0.9],
    output_path="./output/final_mix.wav"
)
```

### Example 4: Audio to Sheet Music

```python
# Extract melody from vocals to MIDI
midi_result = extract_melody_to_midi(
    audio_path="./output/vocals.wav",
    output_midi_path="./output/melody.mid",
    method="basic-pitch"
)

# Refine the MIDI for better notation
refined = refine_midi(
    midi_path="./output/melody.mid",
    output_path="./output/melody_refined.mid",
    operations=["quantize", "remove_short_notes", "smooth_velocities"],
    quantize_grid="16th",
    transpose=2
)

# Export to PDF sheet music
notation = export_notation(
    midi_path="./output/melody_refined.mid",
    output_path="./output/sheet_music.pdf",
    output_format="pdf",
    title="Extracted Melody",
    composer="AI Transcription",
    key_signature="C",
    time_signature="4/4",
    tempo=120.0
)
```

## Directory Structure

```
.
├── Dockerfile           # Docker image configuration
├── docker-compose.yml   # Docker Compose configuration
├── requirements.txt     # Python dependencies
├── server.py            # MCP server implementation
├── test_server.py       # Validation test script
├── mcp-config.json      # Example MCP client configuration
├── tools/               # Tool implementations
│   ├── __init__.py      # Tool registration
│   ├── separation.py    # Audio separation
│   ├── analysis.py      # Audio analysis
│   ├── midi_extraction.py  # Audio to MIDI conversion
│   ├── midi_refinement.py  # MIDI cleanup
│   ├── notation.py      # Music notation export
│   ├── synthesis.py     # MIDI to audio synthesis
│   ├── mixing.py        # Layer mixing and replacement
│   └── effects.py       # Audio effects
├── .dockerignore        # Docker build exclusions
├── .gitignore           # Git exclusions
└── README.md            # This file
```

## Testing

Run the validation tests to ensure everything is configured correctly:

```bash
python3 test_server.py
```

This will verify:
- Server syntax and structure
- All 9 tools are properly defined
- MCP decorators are present
- Dockerfile configuration
- Python dependencies

## GPU Acceleration

The server automatically detects and uses NVIDIA GPUs when available via CUDA. This significantly speeds up:
- Audio separation (Demucs model inference)
- Audio loading and processing (PyTorch operations)

To check if GPU is being used, look for `"device": "cuda:0"` in the tool responses.

## Performance Notes

- **Demucs separation**: On a modern mid-range NVIDIA GPU (for example, an RTX 30-series card), ~30-60 seconds per 3-minute song. Performance will vary significantly based on GPU model, CPU, storage, driver/CUDA versions, and load; expect slower runtimes on older or lower-end hardware and on CPU-only setups.
- **Audio analysis**: ~5-10 seconds per file on typical developer hardware
- **Effect application**: Near real-time on typical developer hardware
- **MIDI synthesis**: Depends on MIDI complexity and synthesis settings

## Troubleshooting

### Permission Denied Errors

The container runs as root internally but automatically changes ownership of output files to match your user. This ensures:
- Cache directories are writable (Numba, Librosa, HuggingFace models)
- Output files are owned by your user account

**For docker-compose:**
```bash
export UID=$(id -u)
export GID=$(id -g)
docker-compose up
```

**For docker run:**
```bash
docker run -e HOST_UID=$(id -u) -e HOST_GID=$(id -g) -i -v $(pwd):/workspace -w /workspace ...
```

**For MCP client:**
Update the HOST_UID and HOST_GID values in your MCP config to match your user (run `id -u` and `id -g` to find these values).

If you still encounter permission issues:
1. Ensure the mounted workspace directory is readable: `ls -la /path/to/workspace`
2. Check Docker has permission to mount the directory
3. On SELinux systems, you may need to add `:z` or `:Z` to volume mounts: `-v $(pwd):/workspace:z`

**Note:** The entire working directory is mounted at `/workspace`, eliminating the need to pre-create input/output directories. The container will create them with the correct ownership automatically.

### GPU Not Detected

If running with `--gpus all` but GPU is not detected:
1. Ensure nvidia-docker2 is installed
2. Check `nvidia-smi` works on host
3. Verify CUDA version compatibility

### Out of Memory

For large audio files:
- Process in chunks
- Use CPU mode for analysis
- Reduce batch size in Demucs

### Audio Quality Issues

- Ensure input files are high quality (>128kbps, 44.1kHz+)
- Use WAV or FLAC for lossless processing
- Check normalization settings

## License

This project is provided as-is for audio processing tasks.

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style
- Tools maintain error handling patterns
- Docker image remains optimized