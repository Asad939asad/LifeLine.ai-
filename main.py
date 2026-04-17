from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Depends
from pydantic import BaseModel, EmailStr
import sqlite3
import secrets
import asyncio
import string
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="./.env")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_ADMIN_SECRET = os.environ.get("HF_ADMIN_SECRET")

# 1. Load HF Secrets into the environment
load_dotenv() 

# --- Import your custom AI pipeline modules ---
from YOLO_Detector import crop_ecg_from_bytes
from LLAVA_FineTuned import analyze_ecg_with_pulse 
from MedGamma_FineTuned import analyze_with_medgemma
from GPTNano_Context import generate_clinical_summary 
from Groq_Summary import generate_master_consensus

app = FastAPI(title="Lifeline ECG Vision API")

# ==========================================
# 1. DATABASE SETUP (SQLite)
# ==========================================
DB_FILE = "/data/lifeline_api.db"

def init_db():
    """Creates the API key table if it doesn't exist yet."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            key TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# Run DB initialization when the server starts
@app.on_event("startup")
def startup_event():
    print("Initializing Database...")
    init_db()

def verify_api_key(api_key: str) -> bool:
    """Checks if the provided API key exists in the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT email FROM api_keys WHERE key = ?", (api_key,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

# ==========================================
# 2. KEY GENERATION ENDPOINT
# ==========================================
class KeyRequest(BaseModel):
    email: EmailStr

@app.post("/v1/keys/generate")
async def generate_api_key(
    request: KeyRequest,
    admin_secret: str = Header(..., description="Master password to generate keys")
):
    # Check if the person requesting a key is actually authorized
    expected_secret = os.environ.get("HF_ADMIN_SECRET", "super_secret_dev_key")
    if admin_secret != expected_secret:
        raise HTTPException(status_code=403, detail="Not authorized to generate keys.")

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Enforce a maximum of 2 keys per email
    cursor.execute("SELECT COUNT(*) FROM api_keys WHERE email = ?", (request.email,))
    key_count = cursor.fetchone()[0]

    if key_count >= 2:
        conn.close()
        raise HTTPException(
            status_code=400, 
            detail=f"Registration limit reached. The email {request.email} already has the maximum of 2 active API keys."
        )

    # --- NEW LOGIC: Generate 'dasa_' followed by 20 random digits ---
    random_digits = ''.join(secrets.choice(string.digits) for _ in range(20))
    new_key = f"dasa_{random_digits}"
    # ----------------------------------------------------------------

    # Save to SQLite Database
    try:
        cursor.execute("INSERT INTO api_keys (key, email) VALUES (?, ?)", (new_key, request.email))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=500, detail="Database integrity error occurred.")
    
    conn.close()

    return {
        "message": "API key generated successfully. Please save it now, as it cannot be retrieved again.",
        "email": request.email,
        "keys_registered_for_email": key_count + 1,
        "api_key": new_key
    }

# ==========================================
# 3. THE CORE AI PIPELINE ENDPOINT
# ==========================================
@app.get("/")
def read_root():
    return {"message": "Lifeline API is live and running!"}

@app.post("/v1/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    x_api_key: str = Header(None) 
):
    # 1. Dynamic Database Authentication
    if not x_api_key or not verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")

    contents = await file.read()

    try:
        # STEP 1: Vision (YOLO Crop)
        cropped_pil_image = crop_ecg_from_bytes(contents)
        
        # STEP 2: Diagnostics (PULSE and MedGemma running concurrently)
        # 🛠️ CRITICAL FIX: Use .copy() so both AI tasks don't fight over the same image file in memory!
        pulse_task = asyncio.to_thread(analyze_ecg_with_pulse, cropped_pil_image.copy())
        medgemma_task = analyze_with_medgemma(cropped_pil_image.copy())
        
        pulse_analysis, medgemma_analysis = await asyncio.gather(pulse_task, medgemma_task)

        # STEP 3: Reference Gathering (GPT-5-Nano)
        gpt_report = await asyncio.to_thread(
            generate_clinical_summary, 
            pulse_analysis, 
            medgemma_analysis
        )

        # STEP 4: The Final Consensus (Groq 70B)
        final_master_report = await asyncio.to_thread(
            generate_master_consensus,
            pulse_analysis,
            medgemma_analysis,
            gpt_report
        )

        # STEP 5: Return the response
        return {
            "status": "success",
            "filename": file.filename,
            "pipeline_metrics": {
                "yolo_crop_size": f"{cropped_pil_image.width}x{cropped_pil_image.height}px"
            },
            "final_report": final_master_report 
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")