FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including those needed for ML libraries
RUN apt-get update && apt-get install -y \
    # Font packages
    fonts-liberation \
    fonts-dejavu-core \
    # Build tools for ML libraries
    build-essential \
    # OpenCV dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Network tools for health check
    curl \
    # Git for downloading models
    git \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set environment variables for better PyTorch performance
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/.torch
ENV HF_HOME=/app/.huggingface

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for model caching
RUN mkdir -p /app/.torch /app/.huggingface /app/models

# Create non-root user and set permissions
RUN useradd -m -u 1001 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Pre-download YOLO model (optional - can be done at runtime)
# RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
