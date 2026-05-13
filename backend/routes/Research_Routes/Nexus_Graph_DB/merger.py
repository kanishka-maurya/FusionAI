import os
import json
import asyncio
from groq import AsyncGroq

client = AsyncGroq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL_NAME = "llama-3.3-70b-versatile"


async def merge_nodes_with_groq(
    parent_summary,
    child_summary,
    parent_keypoints,
    child_keypoints,
    retries=5
):

    merged_keypoints = list(
        set(parent_keypoints + child_keypoints)
    )

    prompt = f"""
You are a semantic graph compression engine.

Merge the following two summaries into ONE unified summary.

Also deduplicate and compress the key points.

Return STRICT JSON ONLY.

{{
  "summary": "...",
  "key_points": ["...", "..."]
}}

SUMMARY 1:
{parent_summary}

SUMMARY 2:
{child_summary}

KEY POINTS:
{merged_keypoints}
"""

    for attempt in range(retries):

        try:

            response = await client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0.2,
                max_tokens=300,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            if not response:
                raise Exception("Empty Groq response")

            if not response.choices:
                raise Exception("No choices returned")

            content = response.choices[0].message.content

            if not content:
                raise Exception("Empty message content")

            content = content.strip()

            if content.startswith("```"):
                content = content.replace("```json", "")
                content = content.replace("```", "")
                content = content.strip()

            parsed = json.loads(content)

            return (
                parsed.get("summary", ""),
                parsed.get("key_points", [])
            )

        except Exception as e:

            wait_time = 2 ** attempt

            print(
                f"Groq merge retry {attempt+1}: {e}"
            )

            print(f"Waiting {wait_time}s")

            await asyncio.sleep(wait_time)

    print("Groq merge failed permanently")

    return (
        f"{parent_summary}\n{child_summary}",
        merged_keypoints
    )