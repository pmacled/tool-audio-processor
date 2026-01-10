# Audio Processor MCP Server

A Docker-based Model Context Protocol (MCP) server for advanced audio layer manipulation, powered by AI and deep learning.

## Features

This MCP server provides six powerful tools for audio processing:

1. **separate_audio_layers** - Split audio into vocals, drums, bass, and other layers using Demucs
2. **analyze_layer** - Extract musical features including notes, tempo, rhythm, and key using librosa
3. **synthesize_instrument_layer** - Generate new instrument audio from MIDI files
4. **replace_layer** - Swap out specific audio layers in a mix
5. **modify_layer** - Apply effects like pitch shift, time stretch, reverb, and normalization
6. **mix_layers** - Combine multiple audio layers with volume control

## Technology Stack

- **Base Image**: NVIDIA CUDA 12.1.0 with cuDNN 8 on Ubuntu 22.04
- **Python**: 3.12+
- **Deep Learning**: PyTorch, Torchaudio
- **Audio Processing**: Demucs (vocal separation), Librosa (analysis)
- **MCP Framework**: FastMCP
- **MIDI**: Pretty MIDI, Mido

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
# Create necessary directories
mkdir -p input output temp

# Run with docker-compose
docker-compose up
```

### Using Docker Directly

#### With GPU Support

```bash
docker run --gpus all -i -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output audio-processor-mcp
```

#### CPU Only

```bash
docker run -i -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output audio-processor-mcp
```

### MCP Client Configuration

Add this to your MCP client configuration (e.g., Claude Desktop):

```json
{
  "mcpServers": {
    "audio-processor": {
      "command": "docker",
      "args": [
        "run",
        "--rm",
        "-i",
        "--gpus",
        "all",
        "-v",
        "${PWD}/input:/app/input",
        "-v",
        "${PWD}/output:/app/output",
        "audio-processor-mcp"
      ]
    }
  }
}
```

See `mcp-config.json` for a complete example.

## MCP Tools Reference

### 1. separate_audio_layers

Separates audio into individual stems using the Demucs model.

**Parameters:**
- `audio_path` (str): Path to input audio file
- `output_dir` (str, optional): Output directory (default: /app/output)
- `model` (str, optional): Demucs model to use (default: htdemucs)

**Returns:**
```json
{
  "success": true,
  "layers": {
    "vocals": "/app/output/song_vocals.wav",
    "drums": "/app/output/song_drums.wav",
    "bass": "/app/output/song_bass.wav",
    "other": "/app/output/song_other.wav"
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

### 3. synthesize_instrument_layer

Generates audio from MIDI files.

**Parameters:**
- `midi_path` (str): Path to MIDI file
- `instrument` (str, optional): Instrument type (default: piano)
- `output_path` (str, optional): Output path (default: /app/output/synthesized.wav)
- `sample_rate` (int, optional): Sample rate (default: 44100)

**Returns:**
```json
{
  "success": true,
  "output_path": "/app/output/synthesized.wav",
  "duration": 120.5,
  "total_notes": 1500
}
```

### 4. replace_layer

Replaces a specific layer in a mixed audio file.

**Parameters:**
- `original_mix_path` (str): Path to original mixed audio
- `layer_to_replace` (str): Layer to replace (vocals, drums, bass, or other)
- `new_layer_path` (str): Path to new layer audio
- `output_path` (str, optional): Output path (default: /app/output/replaced_mix.wav)

**Returns:**
```json
{
  "success": true,
  "output_path": "/app/output/replaced_mix.wav",
  "replaced_layer": "vocals",
  "duration": 180.5
}
```

### 5. modify_layer

Applies audio effects to a layer.

**Parameters:**
- `audio_path` (str): Path to audio file
- `effect` (str): Effect type - "pitch_shift", "time_stretch", "reverb", "normalize", "fade"
- `output_path` (str, optional): Output path (default: /app/output/modified.wav)
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
  "output_path": "/app/output/modified.wav",
  "effect": "pitch_shift",
  "effect_params": {"steps": 2}
}
```

### 6. mix_layers

Combines multiple audio layers into a single mix.

**Parameters:**
- `layer_paths` (List[str]): List of audio file paths
- `output_path` (str, optional): Output path (default: /app/output/mixed.wav)
- `layer_volumes` (List[float], optional): Volume multipliers for each layer
- `normalize_output` (bool, optional): Normalize output (default: true)

**Returns:**
```json
{
  "success": true,
  "output_path": "/app/output/mixed.wav",
  "num_layers": 4,
  "duration": 180.5
}
```

## Usage Examples

### Example 1: Separate and Analyze

```python
# Separate audio into layers
result = separate_audio_layers(
    audio_path="/app/input/song.wav",
    output_dir="/app/output"
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
    original_mix_path="/app/input/original.wav",
    layer_to_replace="vocals",
    new_layer_path="/app/input/new_vocals.wav",
    output_path="/app/output/new_mix.wav"
)
```

### Example 3: Apply Effects and Mix

```python
# Apply pitch shift to vocals
modified = modify_layer(
    audio_path="/app/output/vocals.wav",
    effect="pitch_shift",
    steps=2,
    output_path="/app/output/vocals_shifted.wav"
)

# Mix all layers with custom volumes
final_mix = mix_layers(
    layer_paths=[
        "/app/output/vocals_shifted.wav",
        "/app/output/drums.wav",
        "/app/output/bass.wav",
        "/app/output/other.wav"
    ],
    layer_volumes=[1.2, 1.0, 0.8, 0.9],
    output_path="/app/output/final_mix.wav"
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
- All 6 tools are properly defined
- MCP decorators are present
- Dockerfile configuration
- Python dependencies

## GPU Acceleration

The server automatically detects and uses NVIDIA GPUs when available via CUDA. This significantly speeds up:
- Audio separation (Demucs model inference)
- Audio loading and processing (PyTorch operations)

To check if GPU is being used, look for `"device": "cuda:0"` in the tool responses.

## Performance Notes

- **Demucs separation**: ~30-60 seconds per 3-minute song on GPU, longer on CPU
- **Audio analysis**: ~5-10 seconds per file
- **Effect application**: Near real-time
- **MIDI synthesis**: Depends on MIDI complexity

## Troubleshooting

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