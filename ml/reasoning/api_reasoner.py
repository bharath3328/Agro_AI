import os
from openai import OpenAI

# Get API key from environment variable or config
api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key:
    # Fallback to config if available
    try:
        from backend.config import settings
        api_key = settings.OPENAI_API_KEY
    except ImportError:
        api_key = ""

client = OpenAI(api_key=api_key) if api_key else None

def generate_ai_advisory(
    disease,
    crop,
    confidence,
    severity
):
    
    if not client or not api_key:
        return "AI advisory feature requires OPENAI_API_KEY to be set in environment variables."

    prompt = f"""
You are an expert agricultural plant pathologist.

Disease detected: {disease}
Crop type: {crop}
Prediction confidence: {confidence:.2f}
Disease severity: {severity}

Provide a detailed, disease-specific response covering:
1. Why this disease occurs
2. Visible symptoms
3. Root cause (pathogen or environmental)
4. Immediate action required (treat / monitor / no action)
5. Treatment recommendations
6. Fertilizer suggestions
7. Preventive measures
8. Expected yield loss if untreated

Respond ONLY for this disease.
Avoid generic explanations.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an agricultural advisory AI."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    return response.choices[0].message.content.strip()
