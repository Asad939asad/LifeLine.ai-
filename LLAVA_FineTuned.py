import modal
import io
import os
import json
from PIL import Image
from dotenv import load_dotenv
load_dotenv(dotenv_path="./.env")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_ADMIN_SECRET = os.environ.get("HF_ADMIN_SECRET")
def analyze_ecg_with_pulse(cropped_image: Image.Image) -> dict:
    """
    Takes a cropped PIL Image, sends it to the deployed Modal PULSE-7B app.
    """
    # 1. Convert to bytes
    byte_arr = io.BytesIO()
    cropped_image.save(byte_arr, format='PNG')
    image_bytes = byte_arr.getvalue()

    try:
        # 2. THE NEW MODAL SYNTAX: Use .from_name() instead of .lookup()
        PulseECGModelClass = modal.Cls.from_name("pulse-ecg-analyzer", "PulseECGModel")
        
        # 3. Call the remote method
        print("Sending image to Modal (PULSE-7B)...")
        raw_json_string = PulseECGModelClass().analyze.remote(image_bytes)
        
        print("Received response from Modal!")
        
        # 4. Strip the markdown formatting that causes 500 errors
        clean_json = raw_json_string.replace("```json", "").replace("```", "").strip()
        
        return json.loads(clean_json)
        
    except json.JSONDecodeError:
        print(f"PULSE-7B returned invalid JSON: {raw_json_string}")
        return {
            "overall_interpretation": "Error",
            "findings": [],
            "summary_report": "PULSE-7B did not return valid JSON."
        }
    except Exception as e:
        print(f"❌ Modal Error: {str(e)}")
        # Return a fallback dictionary so the whole pipeline doesn't crash
        return {
            "overall_interpretation": "Error",
            "findings": [],
            "summary_report": f"Failed to connect to PULSE-7B: {str(e)}"
        }