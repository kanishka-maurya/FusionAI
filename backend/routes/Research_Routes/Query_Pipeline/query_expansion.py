import json
import asyncio
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class QueryExpansionService:

    async def expand_query(self, query: str):

        prompt = f"""
You are an analytical query expansion engine.

User Query:
{query}

Generate:
1. 4 related analytical queries.
2. entities for ALL 5 queries.

Return STRICT JSON:
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

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        text = response.text.strip()

        return json.loads(text)


query_expansion_service = QueryExpansionService()