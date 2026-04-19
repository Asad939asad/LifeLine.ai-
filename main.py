from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Depends, Form
from pydantic import BaseModel, EmailStr
from typing import Optional
import sqlite3
import secrets
import asyncio
import string
import os
import io
from PIL import Image
from dotenv import load_dotenv

load_dotenv(dotenv_path="./.env")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_ADMIN_SECRET = os.environ.get("HF_ADMIN_SECRET")

# --- Import your custom AI pipeline modules ---
from YOLO_Detector import crop_ecg_from_bytes
from LLAVA_FineTuned import analyze_ecg_with_pulse 
from MedGamma_FineTuned import analyze_with_medgemma
from GPTNano_Context import generate_clinical_summary 
from Groq_Summary import generate_master_consensus

# --- Import your new dynamic multimodal modules ---
from GPTNano_Prompt import generate_clinical_summary_2
from GROQ_Prompt import generate_master_consensus_2
from LLAVA_Prompt import query_llava_2

app = FastAPI(title="Lifeline ECG Vision API")

# ==========================================
# 1. DATABASE SETUP (SQLite)
# ==========================================
DB_FILE = "/data/lifeline_api.db"
# DB_FILE = "./Database/lifeline_api.db"
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
    # 1. Make email optional so FastAPI doesn't auto-block missing emails
    email: Optional[EmailStr] = None

@app.post("/v1/keys/generate")
async def generate_api_key(
    request: KeyRequest,
    admin_secret: str = Header(..., description="Master password to generate keys")
):
    # 2. NEW LOGIC: Check if the email was provided
    if not request.email:
        raise HTTPException(status_code=400, detail="Without email, no api can be generated")

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

    random_digits = ''.join(secrets.choice(string.digits) for _ in range(20))
    new_key = f"dasa_{random_digits}"

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
# 2.1. KEY DELETION ENDPOINT
# ==========================================

@app.delete("/v1/keys/delete-oldest")
async def delete_oldest_key(
    request: KeyRequest,
    admin_secret: str = Header(..., description="Master password to manage keys")
):
    # 1. Validate Email
    if not request.email:
        raise HTTPException(status_code=400, detail="Email is required to identify the key to delete.")

    # 2. Authenticate Admin
    expected_secret = os.environ.get("HF_ADMIN_SECRET", "super_secret_dev_key")
    if admin_secret != expected_secret:
        raise HTTPException(status_code=403, detail="Not authorized to delete keys.")

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # 3. Find the oldest key for this email
    # We order by created_at ascending (oldest first) and limit to 1
    cursor.execute("""
        SELECT key FROM api_keys 
        WHERE email = ? 
        ORDER BY created_at ASC 
        LIMIT 1
    """, (request.email,))
    
    row = cursor.fetchone()

    # 4. Handle case where no keys exist
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail=f"No active API keys found for email: {request.email}")

    oldest_key = row[0]

    # 5. Delete the key
    try:
        cursor.execute("DELETE FROM api_keys WHERE key = ?", (oldest_key,))
        conn.commit()
        
        # Count remaining keys to return helpful info
        cursor.execute("SELECT COUNT(*) FROM api_keys WHERE email = ?", (request.email,))
        remaining_count = cursor.fetchone()[0]
    except sqlite3.Error:
        conn.close()
        raise HTTPException(status_code=500, detail="Database error occurred while deleting the key.")

    conn.close()

    return {
        "status": "success",
        "message": "Oldest API key deleted successfully.",
        "email": request.email,
        "deleted_key": oldest_key,
        "keys_remaining": remaining_count
    }

# ==========================================
# 3. THE CORE AI PIPELINE ENDPOINT
# ==========================================
@app.get("/")
def read_root():
    return {"message": "Lifeline API is live and ready to server patients !!!"}

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
        pulse_task = asyncio.to_thread(analyze_ecg_with_pulse, cropped_pil_image.copy())
        medgemma_task = analyze_with_medgemma(cropped_pil_image.copy())
        
        pulse_analysis, medgemma_analysis = await asyncio.gather(pulse_task, medgemma_task)
        print("Pulse Analysis: ", pulse_analysis)
        print("MedGemma Analysis: ", medgemma_analysis)
        # STEP 3: Reference Gathering (GPT)
        gpt_report = await asyncio.to_thread(
            generate_clinical_summary, 
            pulse_analysis, 
            medgemma_analysis
        )

        # STEP 4: The Final Consensus (Groq)
        final_master_report = await asyncio.to_thread(
            generate_master_consensus,
            pulse_analysis,
            medgemma_analysis,
            gpt_report
        )

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

# =======================================================
# 4. THE NEW DYNAMIC MULTIMODAL ENDPOINT
# =======================================================
@app.post("/v1/analyze-dynamic")
async def analyze_dynamic_data(
    prompt: str = Form(...),                      # Required Text Prompt
    x_api_key: str = Header(None),
    context: Optional[str] = Form(None),          # Optional Context
    file: Optional[UploadFile] = File(None)      # Optional Image
):
    """
    A separate pipeline that processes a required prompt, an optional image (YOLO cropped), 
    and optional context through LLaVA_2 -> GPT_2 -> Groq_2.
    """
    # 1. Dynamic Database Authentication
    if not x_api_key or not verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")

    pil_image = None
    crop_metrics = None
    
    # 2. Process Image with YOLO if provided
    if file:
        contents = await file.read()
        try:
            # Applying YOLO Crop to the uploaded image
            pil_image = crop_ecg_from_bytes(contents)
            # crop_metrics = f"{pil_image.width}x{pil_image.height}px"
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"YOLO Processing Error: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format or YOLO failure: {e}")

    try:
        # STEP 1: Process with LLaVA
        llava_analysis = await asyncio.to_thread(query_llava_2, prompt, pil_image)

        # STEP 2: Pass LLaVA output and context to GPT
        gpt_report = await asyncio.to_thread(
            generate_clinical_summary_2, 
            llava_analysis, 
            context
        )

        # STEP 3: Final Consensus with Groq
        final_master_report = await asyncio.to_thread(
            generate_master_consensus_2,
            llava_analysis,
            context,
            gpt_report
        )
        print("llava_analysis", llava_analysis)
        # print("gpt_report", gpt_report)
        # print("final_master_report", final_master_report)
        # 4. Compile the Response payload
        response_payload = {
            "status": "success",
            "modality_used": "Text + Image" if pil_image else "Text Only",
            "final_report": final_master_report 
        }
        
        # Only attach pipeline metrics if YOLO actually ran
        # if crop_metrics:
        #     response_payload["pipeline_metrics"] = {"yolo_crop_size": crop_metrics}

        return response_payload

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")