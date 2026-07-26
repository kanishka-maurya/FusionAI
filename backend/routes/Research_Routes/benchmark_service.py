import asyncio
import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

load_dotenv()

judge_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

JUDGE_TIMEOUT_SECONDS = 60
BENCHMARK_OUTPUT_DIR = Path("data") / "benchmarks"

POSITIVE_METRICS = [
    "correctness",
    "completeness",
    "context_coverage",
    "reasoning_depth",
    "actionability",
    "information_gain",
    "citation_coverage",
]

RISK_METRICS = [
    "hallucination_risk"
]

ALL_METRICS = POSITIVE_METRICS + RISK_METRICS
MAX_TEXT_CHARS = 1200
MAX_EVIDENCE_ITEMS = 5


def _text(value, limit=MAX_TEXT_CHARS):

    if value is None:
        return ""

    text = str(value)

    if len(text) <= limit:
        return text

    return text[:limit] + "..."


def _empty_dominance():

    dominance = {
        metric: {
            "fusionai": 0,
            "tavily": 0,
            "tie": 0,
            "leader": "tie"
        }
        for metric in ALL_METRICS
    }

    dominance["overall_winner"] = {
        "fusionai": 0,
        "tavily": 0,
        "tie": 0,
        "leader": "tie"
    }

    return dominance


def _update_leader(bucket):

    counts = {
        "fusionai": bucket["fusionai"],
        "tavily": bucket["tavily"],
        "tie": bucket["tie"]
    }

    leader = max(
        counts,
        key=counts.get
    )

    top_count = counts[leader]

    if list(counts.values()).count(top_count) > 1:
        bucket["leader"] = "tie"
    else:
        bucket["leader"] = leader


def _metric_winner(metric, fusion_score, tavily_score):

    if fusion_score == tavily_score:
        return "tie"

    if metric in RISK_METRICS:
        return (
            "fusionai"
            if fusion_score < tavily_score
            else "tavily"
        )

    return (
        "fusionai"
        if fusion_score > tavily_score
        else "tavily"
    )


def _update_dominance(dominance, judge_result):

    fusion_scores = judge_result.get(
        "fusionai_scores",
        {}
    )

    tavily_scores = judge_result.get(
        "tavily_scores",
        {}
    )

    metric_winners = {}

    for metric in ALL_METRICS:

        fusion_score = fusion_scores.get(metric)
        tavily_score = tavily_scores.get(metric)

        if fusion_score is None or tavily_score is None:
            winner = "tie"
        else:
            winner = _metric_winner(
                metric,
                fusion_score,
                tavily_score
            )

        metric_winners[metric] = winner
        dominance[metric][winner] += 1
        _update_leader(dominance[metric])

    overall_winner = (
        judge_result.get("overall_winner")
        or "tie"
    ).lower()

    if overall_winner not in {"fusionai", "tavily", "tie"}:
        overall_winner = "tie"

    dominance["overall_winner"][overall_winner] += 1
    _update_leader(dominance["overall_winner"])

    return metric_winners


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


def _fallback_judge(reason):

    return {
        "tavily_scores": {
            metric: 0
            for metric in ALL_METRICS
        },
        "fusionai_scores": {
            metric: 0
            for metric in ALL_METRICS
        },
        "overall_winner": "tie",
        "rationale": (
            "LLM judge unavailable; no reliable comparison "
            f"was produced. Reason: {reason}"
        )
    }


def _normalize_score(value):

    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return 0

    return max(
        0,
        min(10, score)
    )


def _normalize_judge_result(raw):

    result = raw if isinstance(raw, dict) else {}

    normalized = {
        "tavily_scores": {},
        "fusionai_scores": {},
        "overall_winner": (
            result.get("overall_winner")
            or "tie"
        ),
        "rationale": result.get("rationale", "")
    }

    for metric in ALL_METRICS:
        normalized["tavily_scores"][metric] = _normalize_score(
            result
            .get("tavily_scores", {})
            .get(metric)
        )
        normalized["fusionai_scores"][metric] = _normalize_score(
            result
            .get("fusionai_scores", {})
            .get(metric)
        )

    winner = str(normalized["overall_winner"]).lower()

    if winner not in {"fusionai", "tavily", "tie"}:
        winner = "tie"

    normalized["overall_winner"] = winner

    return normalized


async def _fetch_tavily_answer(question):

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        return {
            "answer_text": "",
            "top_result_text": "",
            "source_title": "",
            "source_url": "",
            "supporting_texts": [],
            "error": "TAVILY_API_KEY is not configured"
        }

    def run_search():

        tavily = TavilyClient(
            api_key=api_key
        )

        return tavily.search(
            query=question,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
            include_raw_content=False
        )

    try:

        result = await asyncio.wait_for(
            asyncio.to_thread(run_search),
            timeout=30
        )

        results = result.get("results", [])
        top_result = results[0] if results else {}

        answer = (
            result.get("answer")
            or top_result.get("content")
            or top_result.get("title")
            or ""
        )

        return {
            "answer_text": _text(answer),
            "top_result_text": _text(
                top_result.get("content")
                or top_result.get("title")
                or ""
            ),
            "source_title": _text(
                top_result.get("title"),
                300
            ),
            "source_url": _text(
                top_result.get("url"),
                500
            ),
            "supporting_texts": [
                {
                    "title": _text(
                        item.get("title"),
                        300
                    ),
                    "text": _text(
                        item.get("content")
                    )
                }
                for item in results[:MAX_EVIDENCE_ITEMS]
            ],
            "error": None
        }

    except Exception as e:

        return {
            "answer_text": "",
            "top_result_text": "",
            "source_title": "",
            "source_url": "",
            "supporting_texts": [],
            "error": repr(e)
        }


def _extract_fusion_answer(fusion_response):

    user_view = fusion_response.get(
        "user_view",
        {}
    )

    answer = (
        user_view.get("summary")
        or fusion_response
        .get("critic_mode_analysis", {})
        .get("final_intelligence_summary")
        or ""
    )

    answer_parts = [
        _text(answer)
    ]

    key_findings = user_view.get(
        "key_findings",
        []
    )

    if key_findings:
        answer_parts.append(
            "Key findings: "
            + "; ".join(
                _text(item, 300)
                for item in key_findings[:5]
            )
        )

    recommended_searches = (
        user_view.get("recommended_searches")
        or user_view.get("suggested_searches")
        or []
    )

    if recommended_searches:
        answer_parts.append(
            "Related follow-up searches: "
            + "; ".join(
                _text(item, 160)
                for item in recommended_searches[:6]
            )
        )

    limitations = user_view.get(
        "limitations",
        []
    )

    if limitations:
        answer_parts.append(
            "Limitations: "
            + "; ".join(
                _text(item, 220)
                for item in limitations[:3]
            )
        )

    benchmark_answer = "\n\n".join(
        part
        for part in answer_parts
        if part
    )

    evidence_texts = []

    for ctx in user_view.get("retrieved_evidence", [])[:MAX_EVIDENCE_ITEMS]:

        key_points = []

        for item in ctx.get("key_points", [])[:4]:

            if isinstance(item, dict):
                key_points.append(
                    _text(
                        item.get("text", "")
                    )
                )
            else:
                key_points.append(
                    _text(item)
                )

        evidence_texts.append({
            "query": _text(
                ctx.get("query", ""),
                300
            ),
            "summary": _text(
                ctx.get("summary", "")
            ),
            "key_points": key_points
        })

    return {
        "answer_text": _text(
            benchmark_answer,
            1800
        ),
        "topic_text": _text(
            user_view.get("topic_name"),
            300
        ),
        "similar_topic_texts": [
            _text(
                item.get("topic")
                if isinstance(item, dict)
                else item,
                300
            )
            for item in user_view.get("similar_topics", [])[:10]
        ],
        "evidence_texts": evidence_texts
    }


async def _judge_answers(
    question,
    tavily_payload,
    fusion_payload
):

    prompt = f"""
You are an impartial benchmark judge.

Score BOTH answers independently before deciding a winner.

Question:
{question}

Tavily Answer:
{tavily_payload.get("answer_text", "")}

Tavily Supporting Text:
{json.dumps(tavily_payload.get("supporting_texts", []), indent=2)}

FusionAI Answer:
{fusion_payload.get("answer_text", "")}

FusionAI Evidence Text:
{json.dumps(fusion_payload.get("evidence_texts", []), indent=2)}

Metrics:
- correctness: 1-10
- completeness: 1-10
- context_coverage: 1-10
- reasoning_depth: 1-10
- actionability: 1-10
- hallucination_risk: 1-10 where 10 means highest risk
- information_gain: 0-10. Which answer taught the user more beyond a basic search result?
- citation_coverage: 0-10. How broadly and clearly does the answer integrate evidence?

Return STRICT JSON ONLY:
{{
  "tavily_scores": {{
    "correctness": 0,
    "completeness": 0,
    "context_coverage": 0,
    "reasoning_depth": 0,
    "actionability": 0,
    "hallucination_risk": 0,
    "information_gain": 0,
    "citation_coverage": 0
  }},
  "fusionai_scores": {{
    "correctness": 0,
    "completeness": 0,
    "context_coverage": 0,
    "reasoning_depth": 0,
    "actionability": 0,
    "hallucination_risk": 0,
    "information_gain": 0,
    "citation_coverage": 0
  }},
  "overall_winner": "fusionai | tavily | tie",
  "rationale": ""
}}
"""

    try:

        def run_judge():

            return judge_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict JSON-only benchmark judge. "
                            "Score both systems independently before "
                            "choosing a winner."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                response_format={
                    "type": "json_object"
                }
            )

        response = await asyncio.wait_for(
            asyncio.to_thread(run_judge),
            timeout=JUDGE_TIMEOUT_SECONDS
        )

        text = (
            response
            .choices[0]
            .message
            .content
        )

        parsed = json.loads(
            _strip_json_markdown(text)
        )

        return _normalize_judge_result(parsed)

    except Exception as e:

        print(
            "\n[BENCHMARK JUDGE ERROR]",
            repr(e)
        )

        return _fallback_judge(repr(e))


def _persist_json(path, payload):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


def _persist_benchmark_result(payload):

    BENCHMARK_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    run_id = payload["run_id"]
    output_path = BENCHMARK_OUTPUT_DIR / f"{run_id}.json"

    _persist_json(
        output_path,
        payload
    )

    latest_path = BENCHMARK_OUTPUT_DIR / "latest.json"

    _persist_json(
        latest_path,
        payload
    )

    return str(output_path)


def _dominance_from_results(results):

    dominance = _empty_dominance()

    for item in results:

        judge = item.get(
            "judge",
            {}
        )

        _update_dominance(
            dominance,
            judge
        )

    return dominance


def _load_query_evaluations():

    path = BENCHMARK_OUTPUT_DIR / "query_evaluations.json"

    if not path.exists():

        return {
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": None,
            "judge_llm": "groq/llama-3.3-70b-versatile",
            "metrics": {
                "positive_higher_is_better": POSITIVE_METRICS,
                "risk_lower_is_better": RISK_METRICS
            },
            "results": [],
            "final_dominance": _empty_dominance()
        }

    try:

        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except Exception as e:

        print(
            "\n[EVALUATION JSON READ ERROR]",
            repr(e)
        )

        return {
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": None,
            "judge_llm": "groq/llama-3.3-70b-versatile",
            "metrics": {
                "positive_higher_is_better": POSITIVE_METRICS,
                "risk_lower_is_better": RISK_METRICS
            },
            "results": [],
            "final_dominance": _empty_dominance()
        }


async def evaluate_query_response(
    question,
    fusion_response
):

    BENCHMARK_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"\n[EVALUATION] Evaluating query: {question}"
    )

    tavily_payload = await _fetch_tavily_answer(
        question
    )

    fusion_payload = _extract_fusion_answer(
        fusion_response
    )

    judge_result = await _judge_answers(
        question,
        tavily_payload,
        fusion_payload
    )

    existing = _load_query_evaluations()

    result = {
        "evaluation_id": (
            datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            + "_"
            + uuid4().hex[:8]
        ),
        "created_at": datetime.utcnow().isoformat(),
        "question": question,
        "tavily_answer": tavily_payload,
        "fusionai_answer": fusion_payload,
        "judge": judge_result,
        "metric_winners": {}
    }

    temp_dominance = _empty_dominance()
    result["metric_winners"] = _update_dominance(
        temp_dominance,
        judge_result
    )

    existing.setdefault(
        "results",
        []
    ).append(result)

    existing["updated_at"] = datetime.utcnow().isoformat()
    existing["judge_llm"] = "groq/llama-3.3-70b-versatile"
    existing["metrics"] = {
        "positive_higher_is_better": POSITIVE_METRICS,
        "risk_lower_is_better": RISK_METRICS
    }
    existing["final_dominance"] = _dominance_from_results(
        existing["results"]
    )
    existing["total_evaluations"] = len(
        existing["results"]
    )
    existing["json_path"] = str(
        BENCHMARK_OUTPUT_DIR / "query_evaluations.json"
    )

    _persist_json(
        BENCHMARK_OUTPUT_DIR / "query_evaluations.json",
        existing
    )

    _persist_json(
        BENCHMARK_OUTPUT_DIR / "latest_query_evaluation.json",
        result
    )

    print(
        "\n[EVALUATION] Saved query evaluation JSON"
    )

    return {
        "evaluation_id": result["evaluation_id"],
        "metric_winners": result["metric_winners"],
        "overall_winner": judge_result.get(
            "overall_winner",
            "tie"
        ),
        "json_path": existing["json_path"],
        "latest_json_path": str(
            BENCHMARK_OUTPUT_DIR / "latest_query_evaluation.json"
        ),
        "final_dominance": existing["final_dominance"]
    }


async def run_research_benchmark(
    queries,
    mode="deep_research"
):

    from backend.routes.Research_Routes.Query_Pipeline.query_controller import (
        query_controller
    )

    run_id = (
        datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid4().hex[:8]
    )

    dominance = _empty_dominance()
    results = []

    for index, question in enumerate(queries, start=1):

        print(
            f"\n[BENCHMARK] Running query {index}/{len(queries)}: "
            f"{question}"
        )

        tavily_task = _fetch_tavily_answer(question)
        fusion_task = query_controller.process(
            question,
            mode=mode
        )

        tavily_payload, fusion_response = await asyncio.gather(
            tavily_task,
            fusion_task
        )

        fusion_payload = _extract_fusion_answer(
            fusion_response
        )

        judge_result = await _judge_answers(
            question,
            tavily_payload,
            fusion_payload
        )

        metric_winners = _update_dominance(
            dominance,
            judge_result
        )

        results.append({
            "question": question,
            "tavily_answer": tavily_payload,
            "fusionai_answer": fusion_payload,
            "judge": judge_result,
            "metric_winners": metric_winners,
            "running_dominance": deepcopy(dominance)
        })

    benchmark_payload = {
        "run_id": run_id,
        "created_at": datetime.utcnow().isoformat(),
        "judge_llm": "groq/llama-3.3-70b-versatile",
        "benchmark_queries": queries,
        "metrics": {
            "positive_higher_is_better": POSITIVE_METRICS,
            "risk_lower_is_better": RISK_METRICS
        },
        "results": results,
        "final_dominance": dominance
    }

    benchmark_payload["result_json_path"] = str(
        BENCHMARK_OUTPUT_DIR / f"{run_id}.json"
    )
    benchmark_payload["latest_json_path"] = str(
        BENCHMARK_OUTPUT_DIR / "latest.json"
    )

    _persist_benchmark_result(
        benchmark_payload
    )

    return benchmark_payload
