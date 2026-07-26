import asyncio
import json
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

FINAL_ANSWER_TIMEOUT_SECONDS = 45
MAX_CONTEXTS = 7
MAX_SUMMARY_CHARS = 900
MAX_KEYPOINTS = 5
MAX_RELATED_TOPICS = 8

PIPELINE_DIAGNOSTIC_TERMS = [
    "pipeline",
    "audit",
    "diagnostic",
    "retrieval quality",
    "risk features",
    "ethics features",
    "summary-keypoint",
    "system developers",
    "feature builder",
]


def _truncate(text, limit):

    if not text:
        return ""

    text = str(text)

    if len(text) <= limit:
        return text

    return text[:limit] + "..."


def _strip_json_markdown(text):

    text = (text or "").strip()

    if text.startswith("```json"):
        text = (
            text
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

    return text


def _topic_name(topic):

    if isinstance(topic, dict):
        return topic.get("topic") or "Selected topic"

    if topic:
        return str(topic)

    return "Selected topic"


def _looks_like_pipeline_diagnostic(text):

    lowered = str(text or "").lower()

    return any(
        term in lowered
        for term in PIPELINE_DIAGNOSTIC_TERMS
    )


class GroqFinalAnswerService:

    def _compress_contexts(
        self,
        retrieved_contexts
    ):

        contexts = []

        for index, ctx in enumerate(retrieved_contexts[:MAX_CONTEXTS], start=1):

            contexts.append({
                "evidence_id": index,
                "query": _truncate(
                    ctx.get("query", ""),
                    180
                ),
                "summary": _truncate(
                    ctx.get("summary", ""),
                    MAX_SUMMARY_CHARS
                ),
                "key_points": [
                    _truncate(item, 260)
                    if not isinstance(item, dict)
                    else _truncate(item.get("text", item), 260)
                    for item in ctx.get("key_points", [])[:MAX_KEYPOINTS]
                ]
            })

        return contexts

    def _normalize_related_topics(
        self,
        related_topics
    ):

        topics = []

        for item in (related_topics or [])[:MAX_RELATED_TOPICS]:

            if isinstance(item, dict):
                topic = item.get("topic")
            else:
                topic = item

            if topic:
                topics.append(str(topic))

        return topics

    def _fallback_answer(
        self,
        *,
        query,
        selected_topic,
        related_topics,
        contexts,
        reason
    ):

        evidence_lines = []

        for ctx in contexts[:4]:

            summary = ctx.get("summary", "")

            if summary:
                evidence_lines.append(
                    f"[Evidence {ctx['evidence_id']}] {summary}"
                )

        if evidence_lines:
            answer = (
                f"FusionAI found evidence related to {query}. "
                + " ".join(evidence_lines)
            )
        else:
            answer = (
                "FusionAI could not find enough retrieved evidence to answer "
                f"{query} confidently."
            )

        key_findings = [
            line
            for line in evidence_lines[:4]
        ]

        return {
            "answer": answer,
            "key_findings": key_findings,
            "evidence_used": contexts[:4],
            "recommended_searches": related_topics[:6],
            "limitations": [
                (
                    "Final LLM synthesis was unavailable, so this answer was "
                    f"assembled from retrieved evidence. Reason: {reason}"
                )
            ],
            "selected_topic": _topic_name(selected_topic)
        }

    async def generate_answer(
        self,
        *,
        query,
        selected_topic,
        related_topics,
        expanded_queries,
        retrieved_contexts,
        audit_summary=None
    ):

        print("\n[FINAL ANSWER] Starting direct synthesis")

        contexts = self._compress_contexts(
            retrieved_contexts
        )

        normalized_topics = self._normalize_related_topics(
            related_topics
        )

        if not contexts:
            return self._fallback_answer(
                query=query,
                selected_topic=selected_topic,
                related_topics=normalized_topics,
                contexts=[],
                reason="no retrieved contexts"
            )

        prompt = f"""
You are FusionAI's final research answer writer.

Your job is to answer the user's ORIGINAL QUESTION directly.
Do not discuss internal system stages, scoring, or implementation details.
Use only the retrieved evidence below. If the evidence includes live web evidence, integrate it before older graph evidence.
If evidence is incomplete, say what is missing, but still answer the parts that are supported.
Write a self-contained answer that is more useful than a search snippet.
For definition queries, define the term, explain how it works, and give examples.
For comparison queries, directly compare the concepts in clear dimensions.
For update queries, summarize the most important recent developments and implications.
Integrate evidence explicitly with short markers like [Evidence 1] in every paragraph.
Include related topics as "You can also search for..." suggestions.

Original question:
{query}

Selected topic:
{json.dumps(selected_topic, ensure_ascii=False)}

Expanded queries:
{json.dumps(expanded_queries[:5], ensure_ascii=False)}

Related topics:
{json.dumps(normalized_topics, ensure_ascii=False)}

Retrieved evidence:
{json.dumps(contexts, ensure_ascii=False)}

Return STRICT JSON ONLY:
{{
  "answer": "A direct, complete answer to the original question with [Evidence N] markers in every paragraph. Include definitions, distinctions, examples, and implications when relevant.",
  "key_findings": [
    "Concise evidence-grounded finding"
  ],
  "evidence_used": [
    {{
      "evidence_id": 1,
      "why_it_matters": "How this evidence supports the answer"
    }}
  ],
  "recommended_searches": [
    "related topic or follow-up search"
  ],
  "limitations": [
    "Only include real limits, such as missing freshness or sparse evidence"
  ]
}}
"""

        def call_groq():

            return client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict JSON-only research synthesis "
                            "service. Answer the user's original question "
                            "directly using only supplied evidence."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                response_format={
                    "type": "json_object"
                }
            )

        try:

            response = await asyncio.wait_for(
                asyncio.to_thread(call_groq),
                timeout=FINAL_ANSWER_TIMEOUT_SECONDS
            )

            text = response.choices[0].message.content
            parsed = json.loads(
                _strip_json_markdown(text)
            )

            if not isinstance(parsed, dict) or not parsed.get("answer"):
                raise ValueError("final answer JSON did not include answer")

            if _looks_like_pipeline_diagnostic(parsed.get("answer", "")):
                raise ValueError("final answer described pipeline diagnostics")

            parsed.setdefault(
                "recommended_searches",
                normalized_topics[:6]
            )
            parsed.setdefault(
                "key_findings",
                []
            )
            parsed.setdefault(
                "evidence_used",
                []
            )
            parsed.setdefault(
                "limitations",
                []
            )
            parsed["selected_topic"] = _topic_name(
                selected_topic
            )

            print("\n[FINAL ANSWER] Direct synthesis completed")

            return parsed

        except asyncio.TimeoutError:

            print("\n[FINAL ANSWER TIMEOUT] Direct synthesis timed out")

            return self._fallback_answer(
                query=query,
                selected_topic=selected_topic,
                related_topics=normalized_topics,
                contexts=contexts,
                reason="timeout"
            )

        except Exception as e:

            print(
                "\n[FINAL ANSWER ERROR]",
                repr(e)
            )

            return self._fallback_answer(
                query=query,
                selected_topic=selected_topic,
                related_topics=normalized_topics,
                contexts=contexts,
                reason=repr(e)
            )


groq_final_answer_service = GroqFinalAnswerService()
