import os
from openai import OpenAI
from dotenv import load_dotenv
import time

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Set it in .env or environment.")

client = OpenAI(api_key=API_KEY)


def call_gpt4o(prompt, temperature=0.2, max_tokens=1500, retries=3):
    """
    Generic GPT-4o call wrapper with retry logic.
    """
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert multilingual scientific fact-checking assistant."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI error (attempt {attempt+1}): {e}")
            time.sleep(2)

    return ""
