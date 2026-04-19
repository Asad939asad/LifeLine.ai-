from __future__ import annotations
import os
import json
from groq import Groq
from dotenv import load_dotenv
load_dotenv(dotenv_path="./.env")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_ADMIN_SECRET = os.environ.get("HF_ADMIN_SECRET")
def generate_master_consensus(llava_data: dict, medgemma_data: dict | str, gpt_summary: str) -> str:
    """
    Acts as the 'Chief Medical Officer'. Weighs LLaVA (80%) and MedGemma (20%), 
    reads the GPT summary, and outputs a final, authoritative report.
    """
    # 1. Load the API key from Hugging Face secrets
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    # 2. Format the raw inputs
    raw_evidence = f"""
    Primary Diagnostic Data (LLaVA): {json.dumps(llava_data)}
    Secondary Diagnostic Data (MedGemma): {json.dumps(medgemma_data)}
    Clinical Reference Summary (GPT-5): {gpt_summary}
    """

    # 3. The "Subconscious" Rules (Strict Safeguards)
    system_prompt = """You are the Chief AI Cardiologist. You are reviewing data from two sub-systems to write a final, authoritative clinical report.

    CRITICAL INTERNAL RULES:
    1. You must weigh the 'Primary Diagnostic Data' as 80% accurate and the 'Secondary Diagnostic Data' as 20% accurate. If they conflict, the Primary Data wins.
    2. ABSOLUTE FORBIDDEN BEHAVIOR: You must NEVER mention the names of the AI models (LLaVA, MedGemma, GPT). 
    3. ABSOLUTE FORBIDDEN BEHAVIOR: You must NEVER mention the 80/20 weighting criteria or how you arrived at the conclusion. Do not say "Based on the 80% weight..."
    4. Speak directly to the patient/doctor as a single, unified medical system.

    FORMAT REQUIRED:
    1. Primary Disease Summary (What is the most likely condition based on the weighted evidence).
    2. Clinical Context & Immediate Action (Synthesized from the Clinical Reference Summary).
    3. Do smmarize GPT-5-Nano's summary in the report. But donot explicitely mention it is from GPT-5-Nano.  

    """

    # 4. Call the Groq API
    try:
        print("Synthesizing Report via Groq...")
        
        # Note: I'm using the model name you provided in your snippet. 
        # If 'openai/gpt-oss-20b' throws an error on Groq, change it to 'llama3-70b-8192'
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b", # Using a highly capable model standard to Groq for complex reasoning
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please generate the final report based on this evidence:\n{raw_evidence}"}
            ],
            temperature=0.2, # Keep temperature low to prevent hallucinations
            max_tokens=2048,
            stream=False # We wait for the whole string so we can return it as JSON
        )
        
        print("Groq Master Report complete!")
        return completion.choices[0].message.content

    except Exception as e:
        print(f"Groq failed: {str(e)}")
        return f"Could not generate final consensus due to an AI error: {str(e)}"