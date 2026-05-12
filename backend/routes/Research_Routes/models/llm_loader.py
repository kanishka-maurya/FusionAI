import os
import httpx
import json

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

HEADERS = {
    "Content-Type": "application/json"
}


def build_prompt(chunk: str) -> str:
    return f"""
You are an expert AI analyst.

Summarize the following text into STRICT JSON format.

Rules:
- Be factual
- No hallucination
- Extract key entities
- Identify risks if any
- Keep summary concise (3-5 lines)
- Return ONLY valid JSON
- Do not add markdown formatting

Return format:

{{
  "summary": "...",
  "key_points": ["...", "..."],
  "entities": ["...", "..."],
  "risks": ["...", "..."]
}}

Text:
{chunk}
"""


async def summarize_with_gemini(chunk: str):
    prompt = build_prompt(chunk)

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 512,
            "responseMimeType": "application/json"
        }
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            GEMINI_URL,
            headers=HEADERS,
            json=payload
        )

    if response.status_code != 200:
        print("Gemini API error:")
        print(response.text)
        return None

    data = response.json()

    try:
        text_output = (
            data["candidates"][0]
            ["content"]["parts"][0]
            ["text"]
            .strip()
        )

        parsed = json.loads(text_output)

        return parsed

    except Exception as e:
        print("Invalid JSON from Gemini")
        print(e)
        print(data)

        return None