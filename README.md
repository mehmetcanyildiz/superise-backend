# Photo Enhancer API

A FastAPI-based photo enhancement service that uses AI models (GFPGAN and Real-ESRGAN) to improve image quality.

## Features

- Face Enhancement using GFPGAN
- Background Enhancement using Real-ESRGAN
- Automatic Color Correction
- Face Glow Effect
- Support for multiple image formats (JPEG, PNG)
- Customizable enhancement settings

## Requirements

- Python 3.8
- CUDA-compatible GPU (optional, for faster processing)
- Docker (optional, for containerized deployment)

## Installation

### Option 1: Local Installation

1. Create a Python 3.8 virtual environment:
```bash
conda create -n photo_enhancer python=3.8
conda activate photo_enhancer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download AI models:
- Download GFPGAN model from [here](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth) and place it in `models/GFPGANv1.4.pth`
- Download Real-ESRGAN model from [here](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth) and place it in `models/RealESRGAN_x2plus.pth`

4. Run the server:
```bash
python -m uvicorn app.main:app --reload
```

### Option 2: Docker Installation

1. Build the Docker image:
```bash
docker build -t photo-enhancer .
```

2. Run the container:
```bash
docker run -d -p 8000:8000 --name photo-enhancer photo-enhancer
```

## API Usage

### Enhance Photo Endpoint

`POST /enhance`

Parameters:
- `file`: Image file (multipart/form-data)
- `FaceEnhance`: Enable GFPGAN face enhancement (boolean, optional)
- `BackgroundEnhance`: Enable Real-ESRGAN background enhancement (boolean, optional)
- `AutoColor`: Enable automatic color correction (boolean, optional)
- `FaceGlow`: Enable face glow effect (boolean, optional)

Example using curl:
```bash
# All enhancements enabled
curl -X POST "http://localhost:8000/enhance?FaceEnhance=true&FaceGlow=true&AutoColor=true&BackgroundEnhance=true" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/photo.jpg"

# Only face enhancement
curl -X POST "http://localhost:8000/enhance?FaceEnhance=true" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/photo.jpg"
```

Example response:
```json
{
  "url": "/uploads/enhanced_photo.jpg",
  "message": "Image enhanced successfully",
  "applied_settings": {
    "FaceEnhance": true,
    "FaceGlow": true,
    "AutoColor": true,
    "BackgroundEnhance": true
  }
}
```

### Health Check Endpoint

`GET /health`

Example:
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy"
}
```

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   └── main.py          # Main application code
├── models/              # AI model files
│   ├── GFPGANv1.4.pth
│   └── RealESRGAN_x2plus.pth
├── uploads/             # Uploaded and processed images
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration
└── README.md           # This file
```

## Environment Variables

- `PYTHONPATH`: Set to `/app` in Docker for proper module imports
- `PYTHONUNBUFFERED`: Set to `1` for immediate log output

## Development

1. Clone the repository
2. Create a virtual environment with Python 3.8
3. Install dependencies
4. Download AI models
5. Run the server in development mode with `--reload` flag

## Production Deployment

For production deployment:
1. Use Docker for containerization
2. Configure proper security measures
3. Set up proper storage for uploads
4. Configure CORS settings as needed
5. Set up proper logging
6. Use a production-grade ASGI server

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [GFPGAN](https://github.com/TencentARC/GFPGAN) for face enhancement
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) for background enhancement
