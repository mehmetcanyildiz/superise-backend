from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
from PIL import Image
import numpy as np
import torch
import cv2
from typing import Optional
from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

app = FastAPI(title="Photo Enhancer API")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
UPLOAD_DIR = "uploads"
MODEL_DIR = "models"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Mount the uploads directory
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GFPGAN model
gfpgan_model_path = os.path.join(MODEL_DIR, 'GFPGANv1.4.pth')
gfpgan = None
try:
    if os.path.exists(gfpgan_model_path) and os.path.getsize(gfpgan_model_path) > 0:
        gfpgan = GFPGANer(
            model_path=gfpgan_model_path,
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
            device=device
        )
except Exception as e:
    print(f"Error loading GFPGAN model: {str(e)}")
    print("Please download the model from: https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth")

# Real-ESRGAN model
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
realesrgan_model_path = os.path.join(MODEL_DIR, 'RealESRGAN_x2plus.pth')
upsampler = None
try:
    if os.path.exists(realesrgan_model_path) and os.path.getsize(realesrgan_model_path) > 0:
        upsampler = RealESRGANer(
            scale=2,
            model_path=realesrgan_model_path,
            model=model,
            tile=400,
            tile_pad=40,
            pre_pad=0,
            half=True,
            device=device
        )
except Exception as e:
    print(f"Error loading Real-ESRGAN model: {str(e)}")
    print("Please download the model from: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth")

def enhance_face(img_cv2):
    """
    Enhance facial features using GFPGAN
    """
    if gfpgan is None:
        return img_cv2
        
    try:
        _, _, enhanced = gfpgan.enhance(
            img_cv2,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )
        return enhanced
    except Exception as e:
        print(f"Error in face enhancement: {str(e)}")
        return img_cv2

def add_face_glow(img_cv2):
    """
    Add subtle glow effect
    """
    try:
        # Convert to LAB color space
        lab = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to lightness channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Increase brightness slightly
        l = cv2.add(l, 10)
        
        # Merge channels and convert back
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        # Blend with original
        return cv2.addWeighted(img_cv2, 0.7, enhanced, 0.3, 0)
    except Exception as e:
        print(f"Error in face glow: {str(e)}")
        return img_cv2

def auto_color_correct(img_cv2):
    """
    Automatic color correction using OpenCV
    """
    try:
        # Convert to LAB color space for better color correction
        lab = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Enhance color channels
        a = cv2.convertScaleAbs(a, alpha=1.1, beta=0)
        b = cv2.convertScaleAbs(b, alpha=1.1, beta=0)
        
        # Merge channels and convert back
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        # Adjust contrast
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=5)
        
        return enhanced
    except Exception as e:
        print(f"Error in color correction: {str(e)}")
        return img_cv2

def enhance_background(img_cv2):
    """
    Enhance background using Real-ESRGAN
    """
    if upsampler is None:
        return img_cv2
        
    try:
        return upsampler.enhance(img_cv2)[0]
    except Exception as e:
        print(f"Error in background enhancement: {str(e)}")
        return img_cv2

def enhance_image(img, settings):
    """
    Enhanced image processing using GFPGAN and Real-ESRGAN
    """
    try:
        # Convert PIL Image to cv2 format
        img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Resize if image is too large
        max_size = 2048
        height, width = img_cv2.shape[:2]
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            img_cv2 = cv2.resize(img_cv2, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        if settings.get("FaceEnhance", False):
            img_cv2 = enhance_face(img_cv2)
        
        if settings.get("BackgroundEnhance", False):
            img_cv2 = enhance_background(img_cv2)
        
        if settings.get("AutoColor", False):
            img_cv2 = auto_color_correct(img_cv2)
        
        if settings.get("FaceGlow", False):
            img_cv2 = add_face_glow(img_cv2)
        
        # Convert back to PIL Image
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
    except Exception as e:
        print(f"Error in image enhancement: {str(e)}")
        return img

@app.post("/enhance")
async def enhance_photo(
    file: UploadFile = File(...),
    FaceEnhance: Optional[bool] = Query(False, description="Enhance facial features using GFPGAN"),
    FaceGlow: Optional[bool] = Query(False, description="Add subtle glow to face"),
    AutoColor: Optional[bool] = Query(False, description="Automatic color correction"),
    BackgroundEnhance: Optional[bool] = Query(False, description="Enhance background using Real-ESRGAN")
):
    """
    Enhance a photo with customizable settings using AI models
    """
    try:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process image
        img = Image.open(file_path)
        
        # Create settings dictionary
        settings = {
            "FaceEnhance": FaceEnhance and gfpgan is not None,
            "FaceGlow": FaceGlow,
            "AutoColor": AutoColor,
            "BackgroundEnhance": BackgroundEnhance and upsampler is not None
        }
        
        # Enhance image with settings
        output = enhance_image(img, settings)
        
        # Save enhanced image
        output_filename = f"enhanced_{file.filename}"
        output_path = os.path.join(UPLOAD_DIR, output_filename)
        output.save(output_path, quality=95)
        
        # Return the URL and applied settings
        image_url = f"/uploads/{output_filename}"
        return JSONResponse(content={
            "url": image_url,
            "message": "Image enhanced successfully",
            "applied_settings": settings
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}
