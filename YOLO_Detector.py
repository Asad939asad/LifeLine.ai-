# # YOLO------------------------------------------------------------------------------------------------------------------------------------

import io
import os
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv(dotenv_path="./.env")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_ADMIN_SECRET = os.environ.get("HF_ADMIN_SECRET")

print("Loading YOLOv11s model...")
yolo_model = YOLO("./Yolo Weights/best.pt") 

def crop_ecg_from_bytes(image_bytes: bytes, return_canvas: bool = False, padding: int = 80):
    """
    Safely handles PNGs and adds an exact pixel border (default 80px) to all sides 
    using ImageOps.expand before running YOLO, improving detection accuracy.
    """
    # 1. Load the incoming ECG
    incoming_img = Image.open(io.BytesIO(image_bytes))
    incoming_img = ImageOps.exif_transpose(incoming_img)
    
    # CRITICAL FIX for PNGs: Convert transparent backgrounds to white, not black
    if incoming_img.mode in ('RGBA', 'LA') or (incoming_img.mode == 'P' and 'transparency' in incoming_img.info):
        alpha = incoming_img.convert('RGBA').split()[-1]
        bg = Image.new("RGB", incoming_img.size, (255, 255, 255))
        bg.paste(incoming_img, mask=alpha)
        incoming_img = bg
    else:
        incoming_img = incoming_img.convert("RGB")
        
    inc_w, inc_h = incoming_img.size

    # 2. Add white padding to all sides effortlessly
    canvas = ImageOps.expand(incoming_img, border=padding, fill='white')

    # 3. Run YOLO on the padded canvas
    results = yolo_model.predict(source=canvas, conf=0.80, save=False)
    
    if len(results[0].boxes) == 0:
        # Fallback to original image if padded canvas fails
        results = yolo_model.predict(source=incoming_img, conf=0.80, save=False)
        if len(results[0].boxes) == 0:
            # --- MODIFIED: Returning error instead of raising ValueError ---
            return {"status": "error", "message": "No ECG detected by YOLO"}
        
        box = results[0].boxes[0].xyxy[0].cpu().numpy()
        cropped_image = incoming_img.crop(map(int, box))
        
        return (cropped_image, canvas) if return_canvas else cropped_image

    # 4. Map coordinates back by simply subtracting the padding
    box = results[0].boxes[0].xyxy[0].cpu().numpy()
    
    real_x1 = max(0, int(box[0]) - padding)
    real_y1 = max(0, int(box[1]) - padding)
    real_x2 = min(inc_w, int(box[2]) - padding)
    real_y2 = min(inc_h, int(box[3]) - padding)

    # 5. Crop from the original image (so no quality is lost)
    cropped_image = incoming_img.crop((real_x1, real_y1, real_x2, real_y2))
        
    return (cropped_image, canvas) if return_canvas else cropped_image


# #Input1------------------------------------------------------------------------------------------------------------------------------------------------



# #Input2------------------------------------------------------------------------------------------------------------------------------------------------------------------



# #GPT 5 Nano----------------------------------------------------------------------------------------------------------------------------------------------------------------



# #LLM--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# #Return--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------