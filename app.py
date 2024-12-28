from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import fastapi as _fapi
from fastapi.responses import Response
import schemas as _schemas
import services as _services
import traceback
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# CORS ayarlarını her yerden erişime aç
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tüm domainlere izin verir, sadece belirli domainler için ["https://example.com"]
    allow_credentials=True,
    allow_methods=["*"],  # Tüm HTTP metodlarına izin verir
    allow_headers=["*"],  # Tüm header'lara izin verir
)

@app.middleware("http")
async def add_ngrok_headers(request, call_next):
    response = await call_next(request)
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response

@app.get("/")
def read_root():
    return {"message": "Welcome to AI Photo Enhancer API"}

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
        "mime": "image/jpg",
        "image": encoded_img
    }
    
    return payload