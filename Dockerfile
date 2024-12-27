# syntax=docker/dockerfile:1

# Base stage for both development and production
FROM python:3.8-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p models uploads

# Download AI models
RUN curl -L https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -o models/GFPGANv1.4.pth && \
    curl -L https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -o models/RealESRGAN_x2plus.pth

# Development stage
FROM base as development
ENV ENVIRONMENT=development
# Copy application code
COPY . .
# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
# Command will be overridden by docker-compose
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production
ENV ENVIRONMENT=production
# Copy only necessary files
COPY app app/
# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
# Run with multiple workers for better performance
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
