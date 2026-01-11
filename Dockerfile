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

# Install Python 3.12 and system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    ffmpeg \
    libsndfile1 \
    fluidsynth \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Install pip for Python 3.12 using get-pip.py
RUN curl -fsS https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.12 get-pip.py \
    && rm get-pip.py

# Upgrade pip, setuptools, and wheel
RUN python3.12 -m pip install --upgrade pip setuptools wheel

# Set working directory to /app for container's internal files
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch with CUDA support first
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other Python dependencies
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Create cache directories with proper permissions
RUN mkdir -p /tmp/numba_cache /tmp/librosa_cache /tmp/huggingface /tmp/torch /tmp/cache /tmp/matplotlib \
    && chmod -R 777 /tmp/numba_cache /tmp/librosa_cache /tmp/huggingface /tmp/torch /tmp/cache /tmp/matplotlib

# Expose MCP server port (if needed for stdio, this is optional)
# The MCP server typically uses stdio for communication

# Set the entrypoint to run the MCP server from /app
ENTRYPOINT ["python", "/app/server.py"]
