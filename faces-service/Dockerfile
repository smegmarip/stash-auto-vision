FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ /app/app/

# Create model directory (populated at container startup by bootstrap.py,
# which downloads ONNX weights from HuggingFace into the faces_models_cache
# named volume mounted here).
RUN mkdir -p /app/models

# Download InsightFace model on first run
# This will be done at runtime to avoid bloating the image

# Expose port
EXPOSE 5003

# Run bootstrap (HF model download) before uvicorn. The bootstrap exit
# code is intentionally ignored with `|| true` so that a failed required
# model download surfaces via /faces/health (503) rather than a crash-loop.
CMD ["sh", "-c", "python3 -m app.bootstrap || true && exec uvicorn app.main:app --host 0.0.0.0 --port 5003"]
