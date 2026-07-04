import json
import asyncio
import os

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

QUERY_EXPANSION_TIMEOUT_SECONDS = 35


def _fallback_expansion(query: str):

    return {
        "queries": [query],
        "entities": {
            query: []
        }
    }


class QueryExpansionService:

    async def expand_query(
        self,
        query: str
    ):

        prompt = f"""
You are an analytical query expansion engine.

User Query:
{query}

Generate:
1. 4 related analytical queries.
2. entities for ALL 5 queries.

Return STRICT JSON ONLY.

Format:
{{
    "queries": [
        "original query",
        "expanded query 1",
        "expanded query 2",
        "expanded query 3",
        "expanded query 4"
    ],
    "entities": {{
        "query text": ["entity1", "entity2"]
    }}
}}
"""

        try:

            print(
                "\n[QUERY EXPANSION] Starting Groq expansion"
            )

            def call_groq():

                return client.chat.completions.create(
                    model="llama-3.3-70b-versatile",

                    messages=[
                        {
                            "role": "system",
                            "content":
                            (
                                "You are a JSON-only "
                                "query expansion engine. "
                                "Never return markdown."
                            )
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],

                    temperature=0.3,

                    response_format={
                        "type": "json_object"
                    }
                )

            response = await asyncio.wait_for(
                asyncio.to_thread(call_groq),
                timeout=QUERY_EXPANSION_TIMEOUT_SECONDS
            )

            text = (
                response
                .choices[0]
                .message
                .content
                .strip()
            )

            expanded = json.loads(text)

            if not isinstance(expanded, dict):

                raise ValueError(
                    "Query expansion response was not a JSON object"
                )

            expanded_queries = expanded.get("queries")
            expanded_entities = expanded.get("entities")

            if (
                not isinstance(expanded_queries, list)
                or not expanded_queries
                or not isinstance(expanded_entities, dict)
            ):

                raise ValueError(
                    "Query expansion response missing queries/entities"
                )

            print(
                "\n[QUERY EXPANSION] Completed"
            )

            return expanded

        except asyncio.TimeoutError:

            print(
                "\n[QUERY EXPANSION TIMEOUT] "
                f"Groq did not respond within "
                f"{QUERY_EXPANSION_TIMEOUT_SECONDS}s. "
                "Continuing with fallback expansion."
            )

            return _fallback_expansion(query)

        except Exception as e:

            print(
                "\n[QUERY EXPANSION ERROR]",
                e
            )

            return _fallback_expansion(query)


query_expansion_service = (
    QueryExpansionService()
)
