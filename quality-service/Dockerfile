FROM python:3.12-slim

# Ensure system is up to date and install build tools and required libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libmagic1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Optional: upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app /app/app

# Expose port
EXPOSE 5006

# Run FastAPI with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5006"]
