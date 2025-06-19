from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import base64
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import json
import os
from typing import Dict, Any

app = FastAPI(
    title="Insect Identification API",
    version="1.0.0",
    description="AI-powered insect identification service"
)

# CORS middleware for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your Flutter app in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InsectIdentificationService:
    def __init__(self):
        self.api_key = os.getenv("ROBOFLOW_API_KEY", "NVfp8h9atJEAWzsw1eZ0")
        self.model_id = "insect-identification-rweyy/7"
        self.base_url = "https://detect.roboflow.com"

    def create_annotated_image(self, image: Image.Image, predictions: list) -> str:
        """Create annotated image and return as base64"""
        try:
            draw = ImageDraw.Draw(image)

            try:
                # Try different font paths for Render environment
                font_paths = [
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/System/Library/Fonts/Arial.ttf"
                ]
                font = None
                for font_path in font_paths:
                    try:
                        font = ImageFont.truetype(font_path, 20)
                        break
                    except:
                        continue
                if font is None:
                    font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()

            for pred in predictions:
                center_x = pred['x']
                center_y = pred['y']
                width = pred['width']
                height = pred['height']

                left = center_x - width / 2
                top = center_y - height / 2
                right = center_x + width / 2
                bottom = center_y + height / 2

                # Draw bounding box
                draw.rectangle([left, top, right, bottom], outline="red", width=3)

                # Label
                class_name = pred['class']
                confidence = pred['confidence']
                label = f"{class_name}: {confidence:.2f}"

                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height
