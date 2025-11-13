# HealthOS Whisper Container Worker - October 2025
# Production-ready audio processing with GPU optimization

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

LABEL maintainer="HealthOS Team"
LABEL description="Medical-grade audio processing: Transcription, Diarization, NER, Paralinguistics, Prosody"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libopenblas0 \
    liblapack3 \
    libomp-dev \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements
COPY container_src/requirements.txt .

# Install non-PyTorch dependencies from PyPI first
RUN pip install --no-cache-dir \
    fastapi==0.115.6 \
    uvicorn[standard]==0.32.1 \
    transformers==4.46.3 \
    accelerate==1.1.1 \
    pyannote.audio==3.2.0 \
    librosa==0.10.2 \
    scipy==1.14.1 \
    numpy==1.26.4 \
    pydantic==2.10.3 \
    python-multipart==0.0.17 \
    httpx==0.28.1 \
    spacy==3.8.2 \
    scikit-learn==1.5.2 \
    pandas==2.2.3 \
    soundfile==0.12.1 \
    jiwer==3.0.5 \
    pyctcdecode==0.5.0 \
    huggingface-hub==0.26.2

# Install PyTorch dependencies from their index
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    torchvision==0.20.1

# Copy application code
COPY container_src/app.py .
COPY . /app

# Create non-root user for security
RUN useradd -m -u 1000 healthos && chown -R healthos:healthos /app

# Set environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Pre-download models on startup (optional - can be done via environment)
ENV HF_HOME=/tmp/hf_home

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8787/health || exit 1

# Expose port
EXPOSE 8787

# Switch to non-root user
USER healthos

# Run application
CMD ["python", "app.py"]
