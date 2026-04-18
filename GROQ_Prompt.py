import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path="./.env")

def generate_master_consensus_2(llava_data: dict | str, context: str | None, gpt_summary: str) -> str:
    """
    Acts as the 'Chief Medical Officer'. Synthesizes the primary LLaVA data,
    the user's context, and the GPT summary into one final, authoritative report.
    """
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    # Format the raw inputs
    llava_str = json.dumps(llava_data) if isinstance(llava_data, dict) else str(llava_data)
    user_context = context if context else "No additional context provided."

    raw_evidence = f"""
    Primary Model Data: {llava_str}
    User Context: {user_context}
    Clinical Reference Summary (GPT): {gpt_summary}
    """

    system_prompt = """You are the Chief AI Clinician. You are reviewing data from a primary diagnostic pipeline and a clinical reference summary to write a final, authoritative clinical report.

    CRITICAL INTERNAL RULES:
    1. Integrate the 'Primary Model Data' with the 'User Context'. Use the 'Clinical Reference Summary' to structure your medical advice.
    2. ABSOLUTE FORBIDDEN BEHAVIOR: You must NEVER mention the names of the AI models (e.g., LLaVA, GPT, Groq). 
    3. Speak directly to the patient/doctor as a single, unified medical system.

    FORMAT REQUIRED:
    1. Primary Summary: A clear statement of the findings, combining the visual data and user context.
    2. Clinical Context & Recommended Action: A synthesized plan of action based on the evidence.
    3. Do smmarize GPT-5-Nano's summary in the report. But donot explicitely mention it is from GPT-5-Nano.  

    """

    try:
        print("Synthesizing Master Report via Groq...")
        
        # Note: 'llama3-70b-8192' or 'mixtral-8x7b-32768' are standard Groq models.
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please generate the final report based on this evidence:\n{raw_evidence}"}
            ],
            temperature=0.2, 
            max_tokens=2048,
            stream=False 
        )
        
        return completion.choices[0].message.content

    except Exception as e:
        print(f"Groq failed: {str(e)}")
        return f"Could not generate final consensus due to an AI error: {str(e)}"