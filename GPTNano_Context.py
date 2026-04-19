import os
import json
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
load_dotenv(dotenv_path="./.env")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_ADMIN_SECRET = os.environ.get("HF_ADMIN_SECRET")
def generate_clinical_summary(pulse_data: dict, medgemma_data: dict | str) -> str:
    """
    Takes the raw diagnostic JSON from PULSE and MedGemma, 
    and uses GPT-5-Nano to write a structured medical report.
    """
    # 1. Securely load the GitHub token from Hugging Face secrets
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return "Error: GITHUB_TOKEN environment variable not found."

    endpoint = "https://models.github.ai/inference"
    model_name = "openai/gpt-4.1-mini"
    
    # 2. Initialize the client
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(token),
    )

    # 3. Format the raw data into a string for the LLM to read
    # We remove the model names here to enforce the rule of never mentioning them.
    raw_evidence = f"""
    Primary Diagnostic Data: {json.dumps(pulse_data)}
    Secondary Diagnostic Data: {json.dumps(medgemma_data)}
    """

    # 4. Construct the prompt asking for exactly what you need
    system_prompt = """You are an expert AI medical assistant acting as a single, unified medical system. 
    You are reviewing data from two diagnostic sub-systems to write a final, authoritative clinical report.

    CRITICAL INTERNAL RULES:
    1. You must weigh the 'Primary Diagnostic Data' as 50% accurate and the 'Secondary Diagnostic Data' as 50% accurate.
    2. ABSOLUTE FORBIDDEN BEHAVIOR: You must NEVER mention the names of any AI models or sub-systems.
    3. ABSOLUTE FORBIDDEN BEHAVIOR: You must NEVER mention the 50/50 weighting criteria or how you arrived at the conclusion.
    4. ABSOLUTE FORBIDDEN BEHAVIOR: Do not use chain-of-thought reasoning. 
    5. Speak directly to the patient/doctor confidently based on the synthesized evidence."""

    user_prompt = f"""
    Based on the following ECG findings:
    {raw_evidence}

    Please write a short, professional document containing exactly these four sections. Output ONLY the sections, nothing else:
    1. Disease Description: A brief explanation of the condition found.
    2. Immediate Response: What immediate first-aid or medical action is required right now.
    3. Age Risk Profile: At what ages this condition is most dangerous or prevalent.
    4. Doctor Necessity: Is it strictly necessary to call a doctor or go to the ER, or is it a routine finding?
    """

    # 5. Call GPT-5-Nano (Synchronously)
    try:
        print("Synthesizing final report with GPT-5-Nano...")
        response = client.complete(
            messages=[
                SystemMessage(system_prompt),
                UserMessage(user_prompt),
            ],
            model=model_name,
            temperature=0.1, # Low temperature ensures it sticks strictly to the facts
            max_tokens=800
        )
        print("GPT-5-Nano synthesis complete!")
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"GPT-5-Nano failed: {str(e)}")
        return f"Could not generate summary due to an AI error: {str(e)}"