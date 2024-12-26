from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import cv2
import numpy as np
import io
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def enhance_face(image):
    # Yüz geliştirme işlemi
    enhanced = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
    return enhanced

def add_face_glow(image):
    # Yüz parlaklığı efekti
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2RGB)
    return enhanced

def auto_color_correction(image):
    # Otomatik renk düzeltme
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2RGB)
    return enhanced

def enhance_background(image):
    # Arka plan geliştirme
    enhanced = cv2.detailEnhance(image, sigma_s=12, sigma_r=0.15)
    return enhanced

@app.route('/api/enhance', methods=['POST'])
def enhance_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    enhancement_type = request.form.get('type', 'face_enhance')
    
    # Read and process image
    img = Image.open(file.stream)
    img_array = np.array(img)
    
    # Apply enhancement based on type
    if enhancement_type == 'face_enhance':
        enhanced_img = enhance_face(img_array)
    elif enhancement_type == 'face_glow':
        enhanced_img = add_face_glow(img_array)
    elif enhancement_type == 'auto_color':
        enhanced_img = auto_color_correction(img_array)
    elif enhancement_type == 'background_enhance':
        enhanced_img = enhance_background(img_array)
    else:
        return jsonify({'error': 'Invalid enhancement type'}), 400
    
    # Convert back to PIL Image
    enhanced_pil = Image.fromarray(enhanced_img)
    
    # Save to bytes
    img_io = io.BytesIO()
    enhanced_pil.save(img_io, 'PNG')
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
