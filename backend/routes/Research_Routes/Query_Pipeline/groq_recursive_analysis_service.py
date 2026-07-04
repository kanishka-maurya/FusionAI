import os
import json
import asyncio
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

GROQ_RECURSIVE_TIMEOUT_SECONDS = 45

MAX_CONTEXTS = 5
MAX_SUMMARY_CHARS = 500
MAX_KEYPOINTS = 5
MAX_QUERIES = 5

def _truncate(text, limit):

    if not text:
        return ""

    text = str(text)

    if len(text) <= limit:
        return text

    return text[:limit] + "..."


def _strip_json_markdown(text):

    text = text.strip()

    if text.startswith("```json"):

        text = (
            text
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

    return text


class GroqRecursiveAnalysisService:

    async def _call_json(
        self,
        *,
        label,
        prompt,
        system_message,
        temperature,
        fallback
    ):

        print(
            f"\n[GROQ RECURSIVE] Starting {label}"
        )

        def call_groq():

            return client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                response_format={
                    "type": "json_object"
                }
            )

        try:

            response = await asyncio.wait_for(
                asyncio.to_thread(call_groq),
                timeout=GROQ_RECURSIVE_TIMEOUT_SECONDS
            )

            text = (
                response
                .choices[0]
                .message.content
                .strip()
            )

            print(
                f"\n[GROQ RECURSIVE] Completed {label}"
            )

            return json.loads(
                _strip_json_markdown(text)
            )

        except asyncio.TimeoutError:

            print(
                f"\n[GROQ RECURSIVE TIMEOUT] {label} "
                f"timed out."
            )

            return fallback

        except Exception as e:

            print(
                f"\n[GROQ RECURSIVE ERROR] {label}",
                repr(e)
            )

            return fallback

    def _compress_contexts(
        self,
        retrieved_contexts
    ):

        compressed = []

        for ctx in retrieved_contexts[:MAX_CONTEXTS]:

            compressed.append({

                "query":
                ctx.get(
                    "query",
                    ""
                ),

                "summary":
                (
                    ctx.get(
                        "summary",
                        ""
                    )[:MAX_SUMMARY_CHARS]
                ),

                "key_points":
                (
                    ctx.get(
                        "key_points",
                        []
                    )[:MAX_KEYPOINTS]
                )

            })

        return compressed

    def _compress_queries(
        self,
        expanded_queries
    ):

        if isinstance(
            expanded_queries,
            list
        ):

            return expanded_queries[:MAX_QUERIES]

        return []

    def _clean_temporal_trend(
        self,
        temporal_trend
    ):

        if not temporal_trend:

            return None

        if not isinstance(
            temporal_trend,
            list
        ):

            return temporal_trend

        cleaned = [

            x
            for x in temporal_trend
            if x is not None

        ]

        if len(cleaned) == 0:

            return None

        return cleaned

    async def generate_recursive_analysis(
        self,
        selected_topic,
        temporal_trend,
        expanded_queries,
        retrieved_contexts,
        related_topics
    ):
        contexts = []

        for ctx in retrieved_contexts[:5]:

            contexts.append(
                {
                    "query":
                    ctx.get("query", ""),

                    "summary":
                    (ctx.get("summary", "")[:500]),

                    "key_points":
                    ctx.get("key_points", [])[:3]
                }
            )

        queries = expanded_queries[:5]

        has_temporal = (
            temporal_trend
            and any(x is not None for x in temporal_trend)
        )

        trend_block = ""

        if has_temporal:

            trend_block = (
                "\nTemporal Trend:\n"
                + json.dumps(temporal_trend)
            )

        stage1_prompt = f"""
                      Topic:
                      {selected_topic}

                      {trend_block}

                      Queries:
                      {json.dumps(queries)}

                      Retrieved Context:
                      {json.dumps(contexts)}

                      Generate 3 follow-up questions.

                      Return JSON:

                      {{
                       "topic_understanding":"",
                       "follow_up_questions":[
                             {{
                              "question":"",
                              "why_generated":"",
                              "knowledge_gap":""
                             }}
                            ]
                      }}
                   """

        followup_output = await self._call_json(
            label="follow-up generation",
            prompt=stage1_prompt,
            system_message="You generate recursive follow-up questions.",
            temperature=0.3,
            fallback={
                "topic_understanding":
                "Follow-up generation unavailable.",
                "follow_up_questions":[]
            }
        )

        follow_up_questions = (
            followup_output.get(
                "follow_up_questions",
                []
            )
            if isinstance(followup_output, dict)
            else []
        )

        if not follow_up_questions:

            return {

                "selected_topic":
                selected_topic,

                "similar_topics":
                related_topics[:10],

                "topic_temporal_trend":
                temporal_trend if has_temporal else None,

                "follow_up_generation":
                followup_output,

                "recursive_answers": {
                    "generated_answers": []
                },

                "critic_mode_analysis": {
                    "self_evaluation": (
                        "Recursive follow-up analysis was not generated."
                    ),
                    "reasoning_weaknesses": [],
                    "hallucination_risks": [],
                    "improvement_plan": [],
                    "final_intelligence_summary": (
                        "The retrieved topic and related topic suggestions "
                        "are available, but recursive follow-up answers were "
                        "skipped because no follow-up questions were produced."
                    )
                }
            }

        stage2_prompt = f"""
                        Topic:
                        {selected_topic}

                        Questions:
                        {json.dumps(follow_up_questions)}

                        Context:
                        {json.dumps(contexts)}

                        Answer every question.

                        Return JSON

                        {{
                        "generated_answers":[
                        {{
                           "question":"",
                           "answer":"",
                           "supporting_summary":""
                        }}
                        ]
                        }}
                        """

        answer_output = await self._call_json(
            label="recursive answers",
            prompt=stage2_prompt,
            system_message="Answer only using provided context.",
            temperature=0.2,
            fallback={
                "generated_answers":[]
            }
        )

        stage3_prompt = f"""
                        Evaluate these answers.

                        Questions:
                        {json.dumps(
                          followup_output.get(
                          "follow_up_questions",
                           []
                         )
                        )}

                        Answers:
                        {json.dumps(
                          answer_output.get(
                          "generated_answers",
                           []
                          )
                        )}

                        Return JSON

                        {{
                        "self_evaluation":"",
                        "reasoning_weaknesses":[],
                        "hallucination_risks":[],
                        "improvement_plan":[],
                        "final_intelligence_summary":""
                        }}
                        """

        critic_output = await self._call_json(
            label="critic analysis",
            prompt=stage3_prompt,
            system_message="Critique reasoning quality.",
            temperature=0.2,
            fallback={
                "self_evaluation":"Skipped.",
                "reasoning_weaknesses":[],
                "hallucination_risks":[],
                "improvement_plan":[],
                "final_intelligence_summary":"Unavailable."
            }
        )

        return {

            "selected_topic":
            selected_topic,

            "similar_topics":
            related_topics[:10],

            "topic_temporal_trend":
            temporal_trend if has_temporal else None,

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
