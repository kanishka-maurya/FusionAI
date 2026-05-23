import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


class GroqRecursiveAnalysisService:

    async def generate_recursive_analysis(
        self,
        selected_topic,
        temporal_trend,
        expanded_queries,
        retrieved_contexts,
        related_topics
    ):

        """
        retrieved_contexts format:

        [
            {
                "query": "...",
                "summary": "...",
                "key_points": [...]
            }
        ]
        """

        stage1_prompt = f"""
You are an Explainable Recursive RAG Engine.

The retrieval pipeline already selected the BEST
parent node for each query using semantic retrieval.

You are given:
1. Best selected topic
2. Temporal trend of that topic
3. Five generated analytical queries
4. Best retrieved summary node for each query
5. Key points extracted from those nodes

Your task:
Generate intelligent follow-up questions
that deepen the user's understanding.

You MUST:
- explain WHY each question was generated
- explain WHICH summary/key point triggered it
- explain WHAT knowledge gap was detected
- explain HOW the temporal trend influenced it

------------------------------------------------
BEST SELECTED TOPIC
------------------------------------------------

{selected_topic}

------------------------------------------------
TEMPORAL TREND
------------------------------------------------

{json.dumps(temporal_trend, indent=2)}

------------------------------------------------
EXPANDED QUERIES
------------------------------------------------

{json.dumps(expanded_queries, indent=2)}

------------------------------------------------
BEST RETRIEVED CONTEXTS
------------------------------------------------

{json.dumps(retrieved_contexts, indent=2)}

------------------------------------------------
RETURN STRICT JSON
------------------------------------------------

{{
    "topic_understanding": "",
    "follow_up_questions": [
        {{
            "question": "",
            "why_generated": "",
            "triggered_by": "",
            "knowledge_gap": "",
            "temporal_reasoning": ""
        }}
    ]
}}
"""

        stage1_response = (
            client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an explainable "
                            "recursive RAG engine."
                        )
                    },
                    {
                        "role": "user",
                        "content": stage1_prompt
                    }
                ],
                temperature=0.4
            )
        )

        stage1_text = (
            stage1_response
            .choices[0]
            .message.content
            .strip()
        )

        if stage1_text.startswith("```json"):

            stage1_text = (
                stage1_text
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        followup_output = json.loads(
            stage1_text
        )

        stage2_prompt = f"""
You previously generated follow-up questions.

Now answer them deeply using:
- the retrieved summaries
- key points
- temporal trends
- topic relationships

------------------------------------------------
TOPIC
------------------------------------------------

{selected_topic}

------------------------------------------------
FOLLOW UP QUESTIONS
------------------------------------------------

{json.dumps(followup_output, indent=2)}

------------------------------------------------
RETRIEVED CONTEXTS
------------------------------------------------

{json.dumps(retrieved_contexts, indent=2)}

------------------------------------------------
RETURN STRICT JSON
------------------------------------------------

{{
    "generated_answers": [
        {{
            "question": "",
            "answer": "",
            "supporting_summary": "",
            "supporting_keypoints": []
        }}
    ]
}}
"""

        stage2_response = (
            client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a recursive "
                            "reasoning engine."
                        )
                    },
                    {
                        "role": "user",
                        "content": stage2_prompt
                    }
                ],
                temperature=0.3
            )
        )

        stage2_text = (
            stage2_response
            .choices[0]
            .message.content
            .strip()
        )

        if stage2_text.startswith("```json"):

            stage2_text = (
                stage2_text
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        answer_output = json.loads(
            stage2_text
        )

        stage3_prompt = f"""
You are now operating in CRITIC MODE.

You generated:
1. Follow-up questions
2. Answers to those questions

Now self-evaluate yourself.

Analyze:
- reasoning quality
- missing knowledge
- hallucination risks
- weak reasoning chains
- missing context
- retrieval limitations
- possible improvements

Explain:
- how you would improve yourself
- what additional retrieval could help
- what deeper questions should exist

------------------------------------------------
FOLLOW UP QUESTIONS
------------------------------------------------

{json.dumps(followup_output, indent=2)}

------------------------------------------------
GENERATED ANSWERS
------------------------------------------------

{json.dumps(answer_output, indent=2)}

------------------------------------------------
RETURN STRICT JSON
------------------------------------------------

{{
    "self_evaluation": "",
    "reasoning_weaknesses": [],
    "hallucination_risks": [],
    "missing_knowledge": [],
    "improvement_plan": [],
    "final_intelligence_summary": ""
}}
"""

        stage3_response = (
            client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a self-evaluating "
                            "critic reasoning engine."
                        )
                    },
                    {
                        "role": "user",
                        "content": stage3_prompt
                    }
                ],
                temperature=0.2
            )
        )

        stage3_text = (
            stage3_response
            .choices[0]
            .message.content
            .strip()
        )

        if stage3_text.startswith("```json"):

            stage3_text = (
                stage3_text
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        critic_output = json.loads(
            stage3_text
        )

        return {

            "selected_topic":
            selected_topic,

            "similar_topics":
            related_topics,

            "topic_temporal_trend":
            temporal_trend,

            "follow_up_generation":
            followup_output,

            "recursive_answers":
            answer_output,

            "critic_mode_analysis":
            critic_output
        }


groq_recursive_analysis_service = (
    GroqRecursiveAnalysisService()
)