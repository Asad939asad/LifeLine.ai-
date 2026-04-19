from __future__ import annotations
import os
import json
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv(dotenv_path="./.env")

def generate_clinical_summary_2(llava_data: dict | str, context: str | None) -> str:
    """
    Takes the output from LLaVA and the user's optional context,
    and uses GPT to write a structured clinical summary.
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return "Error: GITHUB_TOKEN environment variable not found."

    endpoint = "https://models.github.ai/inference"
    model_name = "openai/gpt-4o-mini" # Note: Adjusted to standard Azure/GitHub model name

    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(token),
    )

    # Convert LLaVA dict to string if necessary
    llava_str = json.dumps(llava_data) if isinstance(llava_data, dict) else str(llava_data)
    
    # Format the evidence
    raw_evidence = f"LLaVA Vision/Text Output:\n{llava_str}\n"
    if context:
        raw_evidence += f"\nAdditional Context Provided by User:\n{context}\n"

    system_prompt = """You are an expert AI medical assistant. 
    You will be given diagnostic data from a vision-language model, alongside optional clinical context from the user. 
    Do not use chain-of-thought reasoning; directly output a highly factual, structured report."""

    user_prompt = f"""
    Based on the following data:
    {raw_evidence}

    Please write a short, professional document containing exactly these sections:
    1. Finding Description: A brief explanation of the model's findings, incorporating the user context if relevant.
    2. Immediate Response: What immediate medical action or consideration is required.
    3. Risk Profile: Demographic or conditional risk factors related to this finding.
    4. Clinical Necessity: Is routine follow-up sufficient, or is urgent care required?
    """

    try:
        print("Synthesizing using GPTNano_2...")
        response = client.complete(
            messages=[
                SystemMessage(system_prompt),
                UserMessage(user_prompt),
            ],
            model=model_name,
            temperature=0.1,
            max_tokens=800
        )
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"GPT failed: {str(e)}")
        return f"Could not generate summary due to an AI error: {str(e)}"