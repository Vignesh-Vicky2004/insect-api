from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import base64
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import json
import os
import torch
import numpy as np
from typing import Dict, Any, List
from ultralytics import YOLO
import open_clip

# Create FastAPI app instance
app = FastAPI(
    title="Enhanced Insect Identification API",
    version="2.0.0", 
    description="AI-powered insect identification using YOLO11 + BioClip"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EnhancedInsectIdentificationService:
    def __init__(self):
        # YOLO11 model setup
        self.yolo_model = YOLO("yolo11n.pt")  # You can use your trained model
        self.confidence_threshold = 0.20  # 20% threshold
        
        # BioClip model setup
        try:
            self.bioclip_model, _, self.bioclip_preprocess = open_clip.create_model_and_transforms(
                'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
            )
            self.bioclip_tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.bioclip_model = self.bioclip_model.to(self.device)
        except Exception as e:
            print(f"Warning: BioClip model failed to load: {e}")
            self.bioclip_model = None
        
        # Your insect classes
        self.insect_classes = [
            "green leafhopper adult",
            "green leafhopper larva", 
            "leaf folder adult",
            "leaf folder",
            "pink bollworm",
            "stem borer adult",
            "stem borer"
        ]

    def yolo_detect(self, image: Image.Image) -> List[Dict]:
        """Perform YOLO11 detection on image"""
        try:
            # Convert PIL to array for YOLO
            img_array = np.array(image)
            
            # Run YOLO11 detection
            results = self.yolo_model(img_array)[0]
            
            detections = []
            for box in results.boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.yolo_model.names[class_id]
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'x': center_x,
                    'y': center_y,
                    'width': width,
                    'height': height,
                    'source': 'YOLO11'
                })
            
            return detections
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return []

    def bioclip_classify(self, image: Image.Image) -> Dict:
        """Perform BioClip classification on image"""[1][2]
        if self.bioclip_model is None:
            return None
            
        try:
            # Preprocess image for BioClip
            image_input = self.bioclip_preprocess(image).unsqueeze(0).to(self.device)
            
            # Create text prompts for insect classes
            text_prompts = [f"a photo of {insect_class}" for insect_class in self.insect_classes]
            text_input = self.bioclip_tokenizer(text_prompts).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                image_features = self.bioclip_model.encode_image(image_input)
                text_features = self.bioclip_model.encode_text(text_input)
                
                # Calculate similarities
                similarities = (image_features @ text_features.T).softmax(dim=-1)
                
            # Get top prediction
            top_prob, top_idx = similarities[0].max(0)
            predicted_class = self.insect_classes[top_idx.item()]
            confidence = top_prob.item()
            
            return {
                'class': predicted_class,
                'confidence': confidence,
                'source': 'BioClip',
                'all_predictions': [
                    {'class': cls, 'confidence': float(sim)} 
                    for cls, sim in zip(self.insect_classes, similarities[0])
                ]
            }
        except Exception as e:
            print(f"BioClip classification error: {e}")
            return None

    def create_annotated_image(self, image: Image.Image, predictions: list) -> str:
        """Create annotated image with detections"""
        try:
            draw = ImageDraw.Draw(image)
            
            # Try to load font
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
                # Different colors for different sources
                color = "red" if pred['source'] == 'YOLO11' else "blue"
                
                if 'x' in pred and 'y' in pred:  # YOLO detection with bounding box
                    center_x = pred['x']
                    center_y = pred['y'] 
                    width = pred['width']
                    height = pred['height']
                    
                    left = center_x - width / 2
                    top = center_y - height / 2
                    right = center_x + width / 2
                    bottom = center_y + height / 2
                    
                    # Draw bounding box
                    draw.rectangle([left, top, right, bottom], outline=color, width=3)
                    
                    # Label
                    class_name = pred['class']
                    confidence = pred['confidence']
                    source = pred['source']
                    label = f"{source}: {class_name} ({confidence:.2f})"
                    
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    draw.rectangle([left, top - text_height - 10, left + text_width + 10, top], 
                                 fill=color, outline=color)
                    draw.text((left + 5, top - text_height - 5), label, fill="white", font=font)
                else:  # BioClip classification (full image)
                    # Draw label at top of image
                    class_name = pred['class']
                    confidence = pred['confidence']
                    source = pred['source']
                    label = f"{source}: {class_name} ({confidence:.2f})"
                    
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    draw.rectangle([10, 10, 10 + text_width + 10, 10 + text_height + 10], 
                                 fill=color, outline=color)
                    draw.text((15, 15), label, fill="white", font=font)

            # Convert to base64
            buffer = BytesIO()
            image.save(buffer, format='JPEG', quality=95)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return img_str
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating annotation: {str(e)}")

    async def identify_insect(self, image_file: UploadFile) -> Dict[str, Any]:
        """Enhanced insect identification using YOLO11 + BioClip"""[3][4]
        try:
            # Read uploaded image
            image_bytes = await image_file.read()
            image = Image.open(BytesIO(image_bytes))
            
            # First, try YOLO11 detection
            yolo_detections = self.yolo_detect(image)
            
            all_predictions = []
            high_confidence_detections = []
            low_confidence_detections = []
            
            # Process YOLO detections
            for detection in yolo_detections:
                if detection['confidence'] >= self.confidence_threshold:
                    high_confidence_detections.append(detection)
                else:
                    low_confidence_detections.append(detection)
                all_predictions.append(detection)
            
            # If no high confidence detections or we have low confidence ones, use BioClip
            bioclip_result = None
            if not high_confidence_detections or low_confidence_detections:
                bioclip_result = self.bioclip_classify(image)
                if bioclip_result:
                    all_predictions.append(bioclip_result)
            
            # Create annotated image
            annotated_base64 = None
            if all_predictions:
                annotated_base64 = self.create_annotated_image(image.copy(), all_predictions)
            
            # Format response
            formatted_response = {
                "success": True,
                "detection_count": len(all_predictions),
                "yolo_detections": len(yolo_detections),
                "high_confidence_yolo": len(high_confidence_detections),
                "low_confidence_yolo": len(low_confidence_detections),
                "bioclip_used": bioclip_result is not None,
                "predictions": [
                    {
                        "species": pred.get("class"),
                        "confidence": round(pred.get("confidence", 0), 4),
                        "confidence_percentage": round(pred.get("confidence", 0) * 100, 2),
                        "source": pred.get("source"),
                        "position": {
                            "x": pred.get("x"),
                            "y": pred.get("y")
                        } if 'x' in pred else None,
                        "size": {
                            "width": pred.get("width"),
                            "height": pred.get("height")
                        } if 'width' in pred else None
                    }
                    for pred in all_predictions
                ],
                "image_info": {
                    "width": image.width,
                    "height": image.height
                },
                "annotated_image": annotated_base64,
                "model_info": {
                    "yolo_threshold": self.confidence_threshold,
                    "models_used": [pred['source'] for pred in all_predictions]
                }
            }
            
            return formatted_response
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# Initialize service
insect_service = EnhancedInsectIdentificationService()

# Routes
@app.get("/")
async def root():
    return {
        "message": "Enhanced Insect Identification API",
        "status": "running", 
        "version": "2.0.0",
        "models": ["YOLO11", "BioClip"],
        "endpoints": {
            "identify": "/identify (POST)",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    try:
        return {
            "status": "healthy", 
            "service": "enhanced-insect-identification",
            "yolo_loaded": insect_service.yolo_model is not None,
            "bioclip_loaded": insect_service.bioclip_model is not None
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/identify")
async def identify_insect(image: UploadFile = File(...)):
    """Upload an image and get enhanced insect identification results"""[5]
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if image.size and image.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image file too large (max 10MB)")
    
    try:
        result = await insect_service.identify_insect(image)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
