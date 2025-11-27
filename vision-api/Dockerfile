FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    cron \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ /app/app/

# Copy and setup cleanup script
COPY cleanup_downloads.sh /app/cleanup_downloads.sh
RUN chmod +x /app/cleanup_downloads.sh

# Setup cron job for hourly cleanup
RUN echo "0 * * * * /app/cleanup_downloads.sh >> /var/log/cron.log 2>&1" > /etc/cron.d/cleanup-downloads \
    && chmod 0644 /etc/cron.d/cleanup-downloads \
    && crontab /etc/cron.d/cleanup-downloads

# Create download directory
RUN mkdir -p /tmp/downloads

# Expose port
EXPOSE 5010

# Start cron and run FastAPI with uvicorn
CMD cron && uvicorn app.main:app --host 0.0.0.0 --port 5010
