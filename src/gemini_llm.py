import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from config import ENV_PATH


# -----------------------------------
# LOAD ENVIRONMENT VARIABLES
# -----------------------------------

load_dotenv(ENV_PATH)

# If you get a 404 "model not found" error, set GEMINI_MODEL in .env
DEFAULT_MODEL = "gemini-3.5-flash-lite"


# -----------------------------------
# CREATE GEMINI LLM (created once)
# -----------------------------------

@lru_cache(maxsize=1)
def get_llm():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")

    model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2,   # low = more stable medical answers
    )

    print(f"Gemini LLM loaded successfully! ({model_name})")

    return llm


# -----------------------------------
# CLEAN GEMINI RESPONSE
# -----------------------------------

def clean_response(response):

    content = response.content

    # Normal text response
    if isinstance(content, str):
        return content.strip()

    # Structured response (list of parts)
    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, dict):
                if item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        text_parts.append(text)

            elif isinstance(item, str):
                text_parts.append(item)

        return "\n".join(text_parts).strip()

    return str(content)



# -----------------------------------
# ANALYZE MEDICAL IMAGE
# -----------------------------------

IMAGE_PROMPT = """
You are a medical information assistant.

Analyze the uploaded medical image and answer
the user's question.

USER QUESTION:
{question}

IMPORTANT RULES:

1. Identify only information that can reasonably
   be understood from the image.

2. Explain the information in simple language.

3. Do not give a definitive medical diagnosis.

4. Do not prescribe personalized medicines.

5. Do not provide personalized dosage instructions.

6. If the image is unclear or unreadable,
   clearly mention that.

7. If the image shows an emergency situation,
   advise the user to seek immediate professional
   medical help.

8. If the image contains a medicine name,
   explain its general purpose only.

9. If the image contains a medical report,
   explain the visible values and terms clearly,
   but do not diagnose the patient.

10. Answer in the same language style as the
    user's question.

Return ONLY the explanation text.
Do not return JSON.
Do not return SVG.
Do not return metadata.
Do not return fields such as type, extras, or signature.
"""


def analyze_medical_image(
    image_base64,
    question,
    mime_type="image/jpeg",
):

    llm = get_llm()

    message = [
        (
            "human",
            [
                {
                    "type": "text",
                    "text": IMAGE_PROMPT.format(question=question),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_base64}"
                    },
                },
            ],
        )
    ]

    response = llm.invoke(message)

    return clean_response(response)


# -----------------------------------
# TEST GEMINI TEXT
# -----------------------------------

if __name__ == "__main__":

    llm = get_llm()

    response = llm.invoke("What is first aid?")

    print("\nGemini Response:")
    print(clean_response(response))