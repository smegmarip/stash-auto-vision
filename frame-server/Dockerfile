FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# Install system dependencies including cron for cleanup and PyAV build deps
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    cron \
    build-essential \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Install basicsr separately with PEP 517 to bypass hash checking issue
RUN pip3 install --no-cache-dir --use-pep517 basicsr==1.4.2 || \
    pip3 install --no-cache-dir basicsr==1.4.2 --no-deps && \
    pip3 install --no-cache-dir addict future lmdb numpy opencv-python Pillow pyyaml requests scikit-image scipy tb-nightly tqdm yapf

# Install remaining requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ /app/app/

# Copy cleanup script
COPY cleanup_frames.sh /usr/local/bin/cleanup_frames.sh
RUN chmod +x /usr/local/bin/cleanup_frames.sh

# Create temp directory for frame storage and logs
RUN mkdir -p /tmp/frames /var/log

# Setup cron job for frame cleanup (runs every hour)
RUN echo "0 * * * * /usr/local/bin/cleanup_frames.sh" | crontab -

# Expose port
EXPOSE 5001

# Start cron and uvicorn
CMD cron && uvicorn app.main:app --host 0.0.0.0 --port 5001
