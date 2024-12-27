from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import fastapi as _fapi
import schemas as _schemas
import services as _services
import traceback
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS configuration
allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to AI Photo Enhancer API"}


# Endpoint to test the backend
@app.get("/api")
async def root():
    return {"message": "Welcome to the AI Photo Enhancer with FastAPI"}


@app.post("/api/enhance/")
async def enhance_image(enhanceBase: _schemas._EnhanceBase = _fapi.Depends()):
    
    try:
        encoded_img = await _services.enhance(enhanceBase=enhanceBase)
    except Exception as e:
        print(traceback.format_exc())
        return {"message": f"{e.args}"}
    
    payload = {
        "mime" : "image/jpg",
        "image": encoded_img
        }
    
    return payload
