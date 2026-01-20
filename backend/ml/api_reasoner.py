import os
import time
import logging
import json
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key:
    try:
        from backend.config import settings
        api_key = settings.OPENAI_API_KEY
    except ImportError:
        api_key = ""

client = OpenAI(api_key=api_key) if api_key else None

def call_openai_with_retry(func, **kwargs):
    """
    Executes an OpenAI API call with retry logic for rate limits and timeouts.
    """
    max_retries = 3
    base_wait = 22  # Slightly more than the requested 20s
    
    for attempt in range(max_retries):
        try:
            return func(**kwargs)
        except Exception as e:
            error_str = str(e).lower()
            
            # Check for Rate Limit or Timeout
            is_rate_limit = "rate limit" in error_str or "429" in error_str
            is_timeout = "timeout" in error_str or "timed out" in error_str
            
            if is_rate_limit or is_timeout:
                if attempt == max_retries - 1:
                    logger.error(f"Max retries reached. Last error: {e}")
                    raise e
                
                wait_time = base_wait if is_rate_limit else (5 * (attempt + 1))
                if is_rate_limit:
                    # Exponential backoff for subsequent rate limits? 
                    # The error says "try again in 20s", so valid wait is critical.
                    wait_time = 22 + (attempt * 5)
                
                logger.warning(f"OpenAI API Issue ({'Rate Limit' if is_rate_limit else 'Timeout'}). to retry in {wait_time}s. (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                # Other errors (Auth, Bad Request, etc) - fail immediately
                raise e

def generate_ai_advisory(disease, crop, confidence, severity):
    
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

    try:
        response = call_openai_with_retry(
            func=client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an agricultural advisory AI."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Failed to generate advisory: {e}")
        return f"AI Advisory temporarily unavailable due to high traffic. Please try again later. Error: {str(e)[:100]}"


def translate_text(text: str, target_language: str) -> str:
    """
    Translates the given text to the target language using OpenAI.
    """
    if not client or not api_key:
        return text  # Fallback to original text if API not available

    if target_language.lower() == "english":
        return text
        
    # Validation: Don't send empty or very short non-word strings
    if not text or not text.strip() or len(text.strip()) < 2:
        return text

    prompt = f"""
Translate the following text into {target_language}.
Rules:
1. Output ONLY the translation. 
2. Do not include any conversational text like "Here is the translation" or "It seems...".
3. If the text is a single word or short phrase, translate it directly.
4. Maintain technical accuracy.

Text to translate:
{text}
"""

    try:
        response = call_openai_with_retry(
            func=client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a strict translator. You output only the translated text and nothing else."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        # Return original text on failure so the UI works
        return text


def translate_batch(texts: dict, target_language: str) -> dict:
    """
    Translates a dictionary of texts to the target language using OpenAI in a single call.
    Returns a dictionary with the same keys and translated values.
    """
    if not client or not api_key:
        return texts

    if target_language.lower() == "english":
        return texts

    # Filter out empty/None values to save tokens
    valid_texts = {k: v for k, v in texts.items() if v and isinstance(v, str) and len(v.strip()) > 1}
    
    if not valid_texts:
        return texts

    prompt = f"""
    Translate the values of the following JSON object into {target_language}.
    Return valid JSON only. Do not wrap in markdown code blocks.
    
    JSON to translate:
    {json.dumps(valid_texts, ensure_ascii=False)}
    """

    try:
        response = call_openai_with_retry(
            func=client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a translator. return only raw JSON output."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        translated_json = json.loads(response.choices[0].message.content)
        
        # Merge back with original dictionary (maintaining None/empty for skipped keys)
        result = texts.copy()
        result.update(translated_json)
        return result
        
    except Exception as e:
        logger.error(f"Batch translation error: {e}")
        return texts


