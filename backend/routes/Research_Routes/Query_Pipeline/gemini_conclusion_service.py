import os
import json
import asyncio
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

GEMINI_TIMEOUT_SECONDS = 45


def _fallback_conclusion(reason):

    return {
        "overall_assessment": (
            "Gemini audit was skipped because the conclusion "
            f"service did not complete: {reason}"
        ),
        "detected_issues": [],
        "system_improvements": [],
        "retrieval_improvements": [],
        "clustering_improvements": [],
        "summarization_improvements": [],
        "final_conclusion": "The retrieval pipeline continued with fallback audit output."
    }


class GeminiConclusionService:

    async def generate_conclusion(
        self,
        risk_features,
        ethics_features,
        audit_features
    ):

        prompt = f"""
You are an Explainable AI Auditor.

You are analyzing abnormalities detected from a
retrieval augmented generation pipeline.

The pipeline:
- generated embeddings for summaries/key points
- used cross-encoder attention models
- detected semantic abnormalities

FEATURE MEANING:

1. RISK FEATURES
These represent highly similar summaries from
different retrieved nodes using cross-encoder attention.
Very high attention means retrieval redundancy or
abnormal semantic overlap.

2. ETHICS FEATURES
These represent low semantic alignment between
nodes inside the SAME cluster/subtree.
Low attention means inconsistent clustering.

3. AUDIT FEATURES
These represent weak alignment between a node summary
and its own key points.
Low attention means the summary may not be fully
grounded in its evidence.

Your task:
- Analyze these abnormalities
- Explain what they indicate
- Explain how the retrieval system can improve itself
- Explain how clustering/retrieval/summarization quality
  can be improved
- Explain possible risks
- Explain confidence level

RISK FEATURES:
{json.dumps(risk_features, indent=2)}

ETHICS FEATURES:
{json.dumps(ethics_features, indent=2)}

AUDIT FEATURES:
{json.dumps(audit_features, indent=2)}

Return STRICT JSON:

{{
    "overall_assessment": "",
    "detected_issues": [],
    "system_improvements": [],
    "retrieval_improvements": [],
    "clustering_improvements": [],
    "summarization_improvements": [],
    "final_conclusion": ""
}}
"""

        try:

            print(
                "\n[GEMINI CONCLUSION] Starting"
            )

            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.models.generate_content,
                    model="gemini-2.5-flash",
                    contents=prompt
                ),
                timeout=GEMINI_TIMEOUT_SECONDS
            )

            text = response.text.strip()

            if text.startswith("```json"):
                text = (
                    text.replace("```json", "")
                    .replace("```", "")
                    .strip()
                )

            print(
                "\n[GEMINI CONCLUSION] Completed"
            )

            return json.loads(text)

        except asyncio.TimeoutError:

            print(
                "\n[GEMINI CONCLUSION TIMEOUT] "
                f"Gemini did not respond within {GEMINI_TIMEOUT_SECONDS}s"
            )

            return _fallback_conclusion("timeout")

        except Exception as e:

            print(
                "\n[GEMINI CONCLUSION ERROR]",
                repr(e)
            )

            return _fallback_conclusion(repr(e))


gemini_conclusion_service = (
    GeminiConclusionService()
)
