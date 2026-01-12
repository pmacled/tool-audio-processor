FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    LIBROSA_CACHE_DIR=/tmp/librosa_cache \
    HF_HOME=/tmp/huggingface \
    TORCH_HOME=/tmp/torch \
    XDG_CACHE_HOME=/tmp/cache \
    MPLCONFIGDIR=/tmp/matplotlib

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    build-essential \
    gcc \
    g++ \
    ffmpeg \
    libsndfile1 \
    fluidsynth \
    lilypond \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install pip for Python 3.11 using get-pip.py
RUN curl -fsS https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.11 get-pip.py \
    && rm get-pip.py

# Upgrade pip, setuptools, and wheel
RUN python3.11 -m pip install --upgrade pip setuptools wheel

# Set working directory to /app for container's internal files
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch with CUDA support first
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other Python dependencies
RUN pip install -r requirements.txt

# Install audio-separator separately without dependencies to avoid beartype conflict
# audio-separator declares beartype<0.19.0 but works fine with beartype>=0.20.0 (required by fastmcp)
RUN pip install --no-deps audio-separator==0.22.0

# Pre-download Demucs models during build to avoid runtime download delays
RUN python -c "from demucs.pretrained import get_model; \
    print('Downloading mdx model...'); \
    get_model('mdx'); \
    print('mdx model downloaded successfully'); \
    print('Downloading htdemucs model...'); \
    get_model('htdemucs'); \
    print('htdemucs model downloaded successfully'); \
    print('Downloading mdx_extra model...'); \
    get_model('mdx_extra'); \
    print('mdx_extra model downloaded successfully')"

# Copy application code
COPY . .

# Pre-download RoFormer models
RUN python -c "from utils.model_downloads import download_roformer_models; \
    print('Downloading RoFormer models...'); \
    download_roformer_models(); \
    print('RoFormer models downloaded successfully')"

# Pre-download SATB models from Google Drive
RUN python -c "from utils.model_downloads import download_satb_models; \
    print('Downloading SATB models from Google Drive...'); \
    download_satb_models(); \
    print('SATB models downloaded successfully')"

# Create cache directories and model directories with proper permissions
RUN mkdir -p /tmp/numba_cache /tmp/librosa_cache /tmp/huggingface /tmp/torch /tmp/cache /tmp/matplotlib /app/models/roformer /app/models/satb \
    && chmod -R 755 /tmp/numba_cache /tmp/librosa_cache /tmp/huggingface /tmp/torch /tmp/cache /tmp/matplotlib /app/models

# Expose MCP server port (if needed for stdio, this is optional)
# The MCP server typically uses stdio for communication

# Switch to /workspace for runtime (matches docker-compose.yml)
# Application code remains at /app, user files are mounted at /workspace
WORKDIR /workspace

# Set the entrypoint to run the MCP server
ENTRYPOINT ["python", "/app/server.py"]
