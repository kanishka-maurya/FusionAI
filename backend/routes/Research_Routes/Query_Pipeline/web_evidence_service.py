import asyncio
import os

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

WEB_EVIDENCE_TIMEOUT_SECONDS = 20
MAX_WEB_EVIDENCE = 4
MAX_CONTENT_CHARS = 900


def _truncate(text, limit):

    if not text:
        return ""

    text = str(text).strip()

    if len(text) <= limit:
        return text

    return text[:limit] + "..."


def _key_points_from_text(text):

    sentences = [
        item.strip()
        for item in str(text or "").replace("\n", " ").split(".")
        if len(item.strip()) > 40
    ]

    return [
        sentence + "."
        for sentence in sentences[:3]
    ]


class WebEvidenceService:

    async def search(
        self,
        query
    ):

        api_key = os.getenv("TAVILY_API_KEY")

        if not api_key:
            print("\n[WEB EVIDENCE] Skipped: TAVILY_API_KEY missing")
            return []

        print("\n[WEB EVIDENCE] Starting live evidence search")

        def run_search():

            tavily = TavilyClient(
                api_key=api_key
            )

            return tavily.search(
                query=query,
                search_depth="advanced",
                max_results=MAX_WEB_EVIDENCE,
                include_answer=True,
                include_raw_content=False
            )

        try:

            result = await asyncio.wait_for(
                asyncio.to_thread(run_search),
                timeout=WEB_EVIDENCE_TIMEOUT_SECONDS
            )

            evidence = []

            answer = result.get("answer")

            if answer:
                evidence.append({
                    "query": query,
                    "summary": _truncate(
                        answer,
                        MAX_CONTENT_CHARS
                    ),
                    "key_points": _key_points_from_text(answer),
                    "provenance": [{
                        "source": "tavily_answer",
                        "title": "Tavily synthesized answer",
                        "url": ""
                    }],
                    "score_details": {
                        "source": "live_web",
                        "hybrid_score": 0.95
                    }
                })

            for item in result.get("results", [])[:MAX_WEB_EVIDENCE]:

                content = (
                    item.get("content")
                    or item.get("title")
                    or ""
                )

                if not content:
                    continue

                evidence.append({
                    "query": query,
                    "summary": _truncate(
                        content,
                        MAX_CONTENT_CHARS
                    ),
                    "key_points": _key_points_from_text(content),
                    "provenance": [{
                        "source": "tavily",
                        "title": _truncate(
                            item.get("title", ""),
                            180
                        ),
                        "url": item.get("url", "")
                    }],
                    "score_details": {
                        "source": "live_web",
                        "hybrid_score": float(
                            item.get("score")
                            or 0.85
                        )
                    }
                })

            print(
                "\n[WEB EVIDENCE] Completed live evidence search: "
                f"{len(evidence)} items"
            )

            return evidence[:MAX_WEB_EVIDENCE]

        except Exception as e:

            print(
                "\n[WEB EVIDENCE ERROR]",
                repr(e)
            )

            return []


web_evidence_service = WebEvidenceService()
