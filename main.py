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

# Create FastAPI app instance - THIS IS CRITICAL
app = FastAPI(
    title="Insect Identification API",
    version="1.0.0",
    description="AI-powered insect identification service"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InsectIdentificationService:
    def __init__(self):
        self.api_key = os.getenv("ROBOFLOW_API_KEY", "NVfp8h9atJEAWzsw1eZ0")
        self.model_id = "insect-identification-rweyy/9"
        self.base_url = "https://detect.roboflow.com"
    
    def create_annotated_image(self, image: Image.Image, predictions: list) -> str:
        """Create annotated image and return as base64"""
        try:
            draw = ImageDraw.Draw(image)
            
            try:
                font_paths = [
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
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
                text_height = bbox[3] - bbox[1]
                
                draw.rectangle([left, top - text_height - 10, left + text_width + 10, top],
                             fill="red", outline="red")
                draw.text((left + 5, top - text_height - 5), label, fill="white", font=font)
            
            # Convert to base64
            buffer = BytesIO()
            image.save(buffer, format='JPEG', quality=95)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return img_str
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating annotation: {str(e)}")
    
    async def identify_insect(self, image_file: UploadFile) -> Dict[str, Any]:
        """Process image and return identification results"""
        try:
            # Read uploaded image
            image_bytes = await image_file.read()
            img_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # API request to Roboflow
            url = f"{self.base_url}/{self.model_id}"
            params = {
                "api_key": self.api_key,
                "confidence": 0.4,
                "overlap": 0.3,
                "format": "json"
            }
            
            response = requests.post(
                url,
                params=params,
                data=img_b64,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Roboflow API error: {response.text}")
            
            result = response.json()
            predictions = result.get("predictions", [])
            
            # Create annotated image
            annotated_base64 = None
            if predictions:
                original_image = Image.open(BytesIO(image_bytes))
                annotated_base64 = self.create_annotated_image(original_image, predictions)
            
            # Format response
            formatted_response = {
                "success": True,
                "detection_count": len(predictions),
                "predictions": [
                    {
                        "species": pred.get("class"),
                        "confidence": round(pred.get("confidence", 0), 4),
                        "confidence_percentage": round(pred.get("confidence", 0) * 100, 2),
                        "position": {
                            "x": pred.get("x"),
                            "y": pred.get("y")
                        },
                        "size": {
                            "width": pred.get("width"),
                            "height": pred.get("height")
                        },
                        "detection_id": pred.get("detection_id")
                    }
                    for pred in predictions
                ],
                "image_info": {
                    "width": result.get("image", {}).get("width"),
                    "height": result.get("image", {}).get("height")
                },
                "annotated_image": annotated_base64,
                "processing_time": result.get("time", 0)
            }
            
            return formatted_response
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# Initialize service
insect_service = InsectIdentificationService()

# Routes
@app.get("/")
async def root():
    return {
        "message": "Insect Identification API", 
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "identify": "/identify (POST)",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "insect-identification"}

@app.post("/identify")
async def identify_insect(image: UploadFile = File(...)):
    """Upload an image and get insect identification results"""
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if image.size and image.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image file too large (max 10MB)")
    
    try:
        result = await insect_service.identify_insect(image)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
