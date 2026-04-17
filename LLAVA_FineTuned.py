import requests
import io
import json
from PIL import Image

def analyze_ecg_with_pulse(cropped_image: Image.Image) -> dict:
    """
    Takes a cropped PIL Image, sends it to the Mac Mini via ngrok,
    which then forwards it to Modal PULSE-7B.
    """
    # 1. Your ngrok URL from your Mac Mini
    NGROK_URL = "https://gallon-shopper-outrank.ngrok-free.dev/v1/analyze-llava"

    # 2. Convert PIL Image to bytes
    byte_arr = io.BytesIO()
    cropped_image.save(byte_arr, format='PNG')
    image_bytes = byte_arr.getvalue()

    try:
        print(f"🚀 Sending image to Mac Mini Bridge via ngrok...")
        
        # 3. Send the file to your Mac Mini Bridge
        # Note: 'file' must match the key in your FastAPI bridge (UploadFile = File(...))
        files = {"file": ("cropped_ecg.png", image_bytes, "image/png")}
        response = requests.post(NGROK_URL, files=files, timeout=300) # Higher timeout for GPU inference
        
        # Raise error for bad status codes (404, 500, etc)
        response.raise_for_status()
        
        bridge_data = response.json()
        
        # Your bridge returns: {"status": "success", "analysis": "raw_json_string"}
        raw_json_string = bridge_data.get("analysis", "")

        print("Received response from Bridge!")
        
        # 4. Strip markdown formatting and parse JSON
        clean_json = raw_json_string.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
        
    except requests.exceptions.RequestException as e:
        print(f"Connection Error (Hugging Face -> ngrok): {str(e)}")
        return {
            "overall_interpretation": "Error",
            "findings": [],
            "summary_report": f"Could not reach Mac Mini bridge: {str(e)}"
        }
    except Exception as e:
        print(f"General Error: {str(e)}")
        return {
            "overall_interpretation": "Error",
            "findings": [],
            "summary_report": f"Pipeline failure: {str(e)}"
        }