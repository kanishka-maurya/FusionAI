import os
import json
import re
from crewai import LLM
import config


# ── Validate API Key ───────────────────────────────────────────────
llm_api_key = os.getenv("GROQ_API_KEY")
if not llm_api_key:
    raise RuntimeError("GROQ_API_KEY environment variable not set")


# ── Initialize LLM ────────────────────────────────────────────────
llm = LLM(
    model=config.MODEL,
    temperature=config.TEMPERATURE,
    max_tokens=config.MAX_TOKENS,
    api_key=llm_api_key
)


# ── JSON Extraction Helper ────────────────────────────────────────
def _extract_json(text: str) -> dict:
    """
    Cleans LLM output and extracts valid JSON.
    Handles markdown code blocks and extra text safely.
    """
    try:
        # Remove ```json ... ``` or ``` ... ```
        cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())

        # Extract JSON object using regex (important 🔥)
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in response")

        json_str = match.group(0)
        return json.loads(json_str)

    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {e}\nRaw response:\n{text}")


# ── Prompt Templates ──────────────────────────────────────────────
ROADMAP_SYSTEM = """
You are FusionAI's curriculum architect.
Your job is to generate precise, structured learning roadmaps for AI and technology topics.
Always return ONLY valid JSON. No markdown. No explanation. No preamble.
"""

ROADMAP_PROMPT = """
Generate a complete learning roadmap for: "{topic}"
User level: {level}

Return ONLY a valid JSON object with this EXACT structure:
{{
  "title": "Clear roadmap title",
  "description": "One sentence describing what the learner will achieve",
  "nodes": [
    {{
      "id": "node_1",
      "title": "Topic Name",
      "type": "required",
      "level": "beginner",
      "dependencies": [],
      "position": {{ "x": 400, "y": 50 }}
    }}
  ]
}}

STRICT RULES:
1. Generate exactly 8-12 nodes
2. Node IDs must be "node_1", "node_2", etc — sequential integers
3. dependencies is an array of node IDs that must be done first. First nodes always have []
4. Position nodes in a vertical tree layout:
   - Root node: x=400, y=50
   - Each level down: increase y by 160
   - Parallel nodes (same level): space x by 300 apart, centered around 400
5. type must be one of: "required", "optional", "project"
6. level must be one of: "beginner", "intermediate", "advanced"
7. Return ONLY the JSON object. Absolutely no text outside the JSON.
"""


# ── Main Function ────────────────────────────────────────────────
def generate_roadmap_json(topic: str, level: str) -> dict:
    """
    Generates a structured roadmap JSON using LLM.
    """
    prompt = ROADMAP_SYSTEM + "\n" + ROADMAP_PROMPT.format(
        topic=topic,
        level=level
    )

    try:
        response = llm.call(prompt)

        return _extract_json(response)

    except Exception as e:
        print("❌ Error generating roadmap:", str(e))
        raise

# ── Node Content Generation ───────────────────────────────────────────────────
 
CONTENT_SYSTEM = """
You are FusionAI, an expert AI/ML educator.
Generate deep, practical educational content for a single learning topic.
Always return ONLY valid JSON. No markdown. No explanation. No preamble.
"""
 
CONTENT_PROMPT = """
Generate comprehensive educational content for this topic.
 
Roadmap Goal: "{roadmap_title}"
Current Topic: "{node_title}"
User Level: {level}
Prerequisites already learned: {prerequisites}
 
Return ONLY a valid JSON object with this EXACT structure:
{{
  "summary": "2-3 sentence overview of this topic",
  "estimated_time": "X weeks",
  "what_you_will_learn": ["outcome 1", "outcome 2", "outcome 3", "outcome 4"],
  "topics": [
    {{
      "title": "Sub-topic name",
      "explanation": "4-5 paragraphs. Be concrete. Use analogies. Assume user knows the prerequisites.",
      "code_example": "working code snippet if applicable, else null",
      "key_takeaway": "The single most important insight from this sub-topic"
    }}
  ],
  "common_misconceptions": [
    "Misconception 1 and why it is wrong",
    "Misconception 2 and why it is wrong"
  ],
  "resources": [
    {{ "type": "video", "title": "Resource name", "url": "#" }},
    {{ "type": "article", "title": "Resource name", "url": "#" }}
  ],
  "practice_questions": [
    {{
      "question": "Question text",
      "hint": "A Socratic nudge without giving the answer",
      "answer": "Complete answer"
    }}
  ]
}}
 
RULES:
- Generate exactly 3-5 sub-topics inside "topics"
- Explanations must be genuinely educational, not just definitions
- code_example must be valid runnable code or null (not a placeholder)
- Generate exactly 2-3 practice questions
- Return ONLY the JSON object
"""
 
 
async def generate_node_content(
    node_title: str,
    roadmap_title: str,
    level: str,
    prerequisites: list[str]
) -> dict:
    prereq_str = ", ".join(prerequisites) if prerequisites else "None — this is the starting point"
    prompt = CONTENT_PROMPT.format(
        roadmap_title=roadmap_title,
        node_title=node_title,
        level=level,
        prerequisites=prereq_str
    )
    response = llm.call(prompt)
    return _extract_json(response)


# ── Run Script ───────────────────────────────────────────────────
if __name__ == "__main__":
    topic = "machine learning"
    level = "beginner"

    result = generate_roadmap_json(topic, level)

    print("\n✅ Generated Roadmap:\n")
    print(json.dumps(result, indent=2))


