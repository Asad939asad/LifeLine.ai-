
# # YOLO------------------------------------------------------------------------------------------------------------------------------------

import io
import os
from PIL import Image
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv(dotenv_path="./.env")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_ADMIN_SECRET = os.environ.get("HF_ADMIN_SECRET")

print("Loading YOLOv11s model...")
yolo_model = YOLO("./Yolo Weights/best.pt") 

def crop_ecg_from_bytes(image_bytes: bytes, margin_px: int = 80) -> Image.Image:
    """
    Takes raw bytes, finds the ECG paper, and returns a cropped PIL Image.
    margin_px: Number of pixels to expand the crop box to prevent cutting off edges.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_width, img_height = image.size # Get the full size of the original photo
    
    results = yolo_model.predict(source=image, conf=0.25, save=False)
    
    if len(results[0].boxes) == 0:
        raise ValueError("YOLO could not detect an ECG in this image.")
        
    # Extract original YOLO coordinates
    box = results[0].boxes[0].xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)
    
    # --- NEW LOGIC: Expand the box by the margin ---
    # We use max() and min() to ensure we don't accidentally try to crop 
    # outside the actual boundaries of the original photo.
    x1 = max(0, x1 - margin_px)
    y1 = max(0, y1 - margin_px)
    x2 = min(img_width, x2 + margin_px)
    y2 = min(img_height, y2 + margin_px)
    # -----------------------------------------------
    
    # Crop the PIL image mathematically with the new expanded box
    cropped_image = image.crop((x1, y1, x2, y2))
        
    return cropped_image

# #Input1------------------------------------------------------------------------------------------------------------------------------------------------



# #Input2------------------------------------------------------------------------------------------------------------------------------------------------------------------



# #GPT 5 Nano----------------------------------------------------------------------------------------------------------------------------------------------------------------



# #LLM--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# #Return--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



