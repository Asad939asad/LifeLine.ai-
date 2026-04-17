import requests
import io
import json
from PIL import Image
from typing import Optional

def query_llava_2(prompt: str, image: Optional[Image.Image] = None) -> dict:
    """
    Takes a text prompt and an optional PIL Image.
    Routes to the Mac Mini via ngrok for LLaVA/PULSE-7B processing.
    """
    NGROK_URL = "https://gallon-shopper-outrank.ngrok-free.dev/v1/analyze-dynamic-llava"

    # 1. Prepare the text payload
    data_payload = {
        "prompt": prompt
    }

    files_payload = None

    # 2. Conditionally prepare the image if provided
    if image is not None:
        byte_arr = io.BytesIO()
        # Convert to RGB to strip alpha channels, avoiding formatting issues
        image_rgb = image.convert("RGB")
        image_rgb.save(byte_arr, format='PNG')
        image_bytes = byte_arr.getvalue()
        
        # 'file' must match the parameter name in your FastAPI backend
        files_payload = {"file": ("input_image.png", image_bytes, "image/png")}

    try:
        modality = "Text + Image" if image else "Text-Only"
        print(f"Sending {modality} request to Bridge...")
        
        # 3. Send request (requests library handles the multipart formatting automatically)
        response = requests.post(
            NGROK_URL, 
            data=data_payload,    # Sends the prompt
            files=files_payload,  # Sends the image (or None)
            timeout=300
        )
        
        response.raise_for_status()
        bridge_data = response.json()
        
        # 4. Extract and clean the response
        raw_response = bridge_data.get("analysis", "")
        
        # Try to parse as JSON (if you asked for JSON formatting)
        clean_json = raw_response.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(clean_json)
        except json.JSONDecodeError:
            # Fallback if the model returns plain text instead of JSON
            return {"raw_text_response": raw_response}
            
    except requests.exceptions.RequestException as e:
        print(f"Connection Error: {str(e)}")
        return {"error": f"Could not reach Mac Mini bridge: {str(e)}"}
    except Exception as e:
        print(f"General Error: {str(e)}")
        return {"error": f"Pipeline failure: {str(e)}"}