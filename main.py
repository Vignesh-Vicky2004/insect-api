import requests
import base64
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import json
import os


def create_annotated_image(original_image_path, predictions):
    """
    Create annotated image manually using detection coordinates
    """
    try:
        # Open original image
        image = Image.open(original_image_path)
        draw = ImageDraw.Draw(image)

        # Try to load a font (fallback to default if not available)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        # Draw bounding boxes and labels
        for pred in predictions:
            # Get coordinates
            center_x = pred['x']
            center_y = pred['y']
            width = pred['width']
            height = pred['height']

            # Calculate bounding box corners
            left = center_x - width / 2
            top = center_y - height / 2
            right = center_x + width / 2
            bottom = center_y + height / 2

            # Draw bounding box (red color)
            draw.rectangle([left, top, right, bottom], outline="red", width=3)

            # Prepare label text
            class_name = pred['class']
            confidence = pred['confidence']
            label = f"{class_name}: {confidence:.2f}"

            # Get text size for background
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Draw label background
            draw.rectangle([left, top - text_height - 10, left + text_width + 10, top],
                           fill="red", outline="red")

            # Draw label text
            draw.text((left + 5, top - text_height - 5), label, fill="white", font=font)

        return image

    except Exception as e:
        print(f"❌ Error creating annotated image: {e}")
        return None


def enhanced_insect_identification():
    print("🚀 Enhanced insect identification with manual annotation...")

    # Configuration
    api_key = "NVfp8h9atJEAWzsw1eZ0"
    model_id = "insect-identification-rweyy/7"
    image_path = "Pink bollworm larva.jpg"

    try:
        # Read and encode image
        with open(image_path, "rb") as f:
            img_bytes = f.read()
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')

        # API request
        url = f"https://detect.roboflow.com/{model_id}"
        params = {
            "api_key": api_key,
            "confidence": 0.4,
            "overlap": 0.3,
            "format": "json"
        }

        response = requests.post(url, params=params, data=img_b64,
                                 headers={"Content-Type": "application/json"})

        if response.status_code == 200:
            result = response.json()

            # Display results
            print("\n" + "=" * 60)
            print("🔍 INSECT IDENTIFICATION RESULTS")
            print("=" * 60)

            predictions = result.get("predictions", [])
            if predictions:
                for i, pred in enumerate(predictions, 1):
                    print(f"\n🐛 Detection {i}:")
                    print(f"   🏷  Species: {pred.get('class')}")
                    print(f"   📊 Confidence: {pred.get('confidence'):.4f} ({pred.get('confidence') * 100:.2f}%)")
                    print(f"   📍 Center: ({pred.get('x'):.1f}, {pred.get('y'):.1f})")
                    print(f"   📏 Size: {pred.get('width')}×{pred.get('height')} pixels")
                    print(f"   🆔 Detection ID: {pred.get('detection_id')}")

                # Create annotated image manually
                print(f"\n🎨 Creating annotated image...")
                annotated_img = create_annotated_image(image_path, predictions)

                if annotated_img:
                    # Save annotated image
                    output_path = "annotated_pink_bollworm.jpg"
                    annotated_img.save(output_path, quality=95)
                    print(f"✅ Annotated image saved as '{output_path}'")

                    # Display image (optional)
                    try:
                        import matplotlib.pyplot as plt
                        plt.figure(figsize=(12, 8))
                        plt.imshow(annotated_img)
                        plt.axis('off')
                        plt.title(f'Pink-Bollworm Detection\nConfidence: {predictions[0]["confidence"] * 100:.1f}%',
                                  fontsize=16, fontweight='bold')
                        plt.tight_layout()
                        plt.show()
                        print("🖼 Image displayed successfully!")
                    except ImportError:
                        print("📱 Install matplotlib to display image: pip install matplotlib")

                    # Show image with default viewer
                    try:
                        annotated_img.show()
                    except:
                        pass

                # Create detection summary
                print(f"\n📋 DETECTION SUMMARY:")
                print(f"   Total detections: {len(predictions)}")
                print(f"   Primary species: {predictions[0]['class']}")
                print(f"   Highest confidence: {max(p['confidence'] for p in predictions) * 100:.1f}%")
                print(f"   Image dimensions: {result['image']['width']}×{result['image']['height']}")

            else:
                print("❌ No insects detected in the image")

            return result

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


# Alternative: Quick annotated detection
def quick_annotated_detection(image_path):
    """
    Quick function that always creates an annotated image
    """
    print(f"🔍 Quick detection for: {image_path}")

    # Your successful detection data (from the output)
    mock_predictions = [{
        "x": 256.0,
        "y": 160.5,
        "width": 326.0,
        "height": 157.0,
        "confidence": 0.8725875616073608,
        "class": "Pink-Bollworm",
        "class_id": 1,
        "detection_id": "a7037edd-8b54-40ba-9a4a-5a112ca1f84c"
    }]

    # Create annotated image
    annotated_img = create_annotated_image(image_path, mock_predictions)

    if annotated_img:
        annotated_img.save("quick_annotated_result.jpg")
        annotated_img.show()
        print("✅ Quick annotated image created and displayed!")
        return annotated_img

    return None


# For deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

    print("🎯 ENHANCED INSECT IDENTIFICATION")
    print("=" * 60)

    # Run enhanced version
    result = enhanced_insect_identification()

    # Alternative: Use your existing successful detection
    print("\n" + "=" * 60)
    print("🚀 CREATING ANNOTATED IMAGE FROM YOUR DETECTION")
    print("=" * 60)
    quick_annotated_detection("Pink bollworm larva.jpg")
