import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Set it in .env or environment.")

client = OpenAI(api_key=API_KEY)


def call_gpt4o(
    user_message: str,
    system_message: str = "You are an expert multilingual scientific fact-checking assistant.",
    temperature: float = 0.2,
    max_tokens: int = 1500,
    retries: int = 3,
):
    """
    GPT-4o call wrapper with retry logic.
    Accepts BOTH system and user messages.
    """
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI error (attempt {attempt+1}): {e}")
            time.sleep(2)

    return ""


# import os
# from openai import OpenAI
# from dotenv import load_dotenv
# import time

# load_dotenv()

# API_KEY = os.getenv("OPENAI_API_KEY")

# if not API_KEY:
#     raise ValueError("OPENAI_API_KEY not found. Set it in .env or environment.")

# client = OpenAI(api_key=API_KEY)


# def call_gpt4o(prompt, temperature=0.2, max_tokens=1500, retries=3):
#     """
#     Generic GPT-4o call wrapper with retry logic.
#     """
#     for attempt in range(retries):
#         try:
#             response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": "You are an expert multilingual scientific fact-checking assistant."
#                     },
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=temperature,
#                 max_tokens=max_tokens
#             )
#             return response.choices[0].message.content.strip()
#         except Exception as e:
#             print(f"OpenAI error (attempt {attempt+1}): {e}")
#             time.sleep(2)

#     return ""
