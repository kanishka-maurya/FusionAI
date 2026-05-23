import json
import asyncio
import os

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


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

            response = await asyncio.to_thread(
                client.chat.completions.create,

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

            text = (
                response
                .choices[0]
                .message
                .content
                .strip()
            )

            return json.loads(text)

        except Exception as e:

            print(
                "\n[QUERY EXPANSION ERROR]",
                e
            )

            return {
                "queries": [query],
                "entities": {
                    query: []
                }
            }


query_expansion_service = (
    QueryExpansionService()
)