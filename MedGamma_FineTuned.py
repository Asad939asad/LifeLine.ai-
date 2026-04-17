import httpx
import io
from PIL import Image
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="./.env")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_ADMIN_SECRET = os.environ.get("HF_ADMIN_SECRET")
async def analyze_with_medgemma(cropped_image: Image.Image) -> dict | str:
    """
    Sends the cropped ECG to MedGemma via HTTP POST.
    Waits exactly 120 seconds before timing out and returning 'No results'.
    """
    url = "https://deductive-carita-galactic.ngrok-free.dev/predict"

    # 1. Convert the PIL Image back to PNG bytes for the HTTP request
    byte_arr = io.BytesIO()
    cropped_image.save(byte_arr, format='PNG')
    image_bytes = byte_arr.getvalue()

    # 2. Replicate the `curl -F "file=@..."` behavior
    files = {'file': ('ecg_crop.png', image_bytes, 'image/png')}

    # 3. Set the strict 2-minute (120 seconds) timeout
    timeout_config = httpx.Timeout(120.0)

    # 4. Make the asynchronous HTTP request
    async with httpx.AsyncClient(timeout=timeout_config) as client:
        try:
            print("Sending cropped image to MedGemma...")
            response = await client.post(url, files=files)
            response.raise_for_status() # Check for 404, 500, etc.
            
            print("Received response from MedGemma!")
            print("MedGemma Response: ", response.json())
            return response.json()
            
        except httpx.ReadTimeout:
            print("MedGemma timed out after 2 minutes.")
            return "No results (Timed out after 2 minutes)"
            
        except Exception as e:
            print(f"MedGemma request failed: {str(e)}")
            return f"No results (Error: {str(e)})"