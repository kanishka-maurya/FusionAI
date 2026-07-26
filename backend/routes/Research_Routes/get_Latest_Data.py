from fastapi import APIRouter
from datetime import datetime, timedelta

import httpx
import asyncio
import json
import hashlib
import xml.etree.ElementTree as ET
import redis.asyncio as redis
import os
from backend.routes.Research_Routes.utils import source_authority_score
from backend.routes.Research_Routes.contracts import (
    QueryRequest,
)

router = APIRouter()

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)

BLOCKED_KEYWORDS = [
    "cheat",
    "hack",
    "crack",
    "exploit",
    "malware",
    "stealer",
    "phishing",
    "keylogger"
]

ZSET_KEY = "ai:raw:current"
SEEN_KEY = "ai:seen_ids"

ACTIVITY_PREFIX = "ai:activity"
RATE_PREFIX = "ai:rate"
QUERY_ACTIVE_KEY = "ai:query:active_count"
QUERY_ACTIVE_TTL_SECONDS = 600

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

latest_live_data = {
    "github": [],
    "papers": [],
    "news": []
}

active_query_count = 0


async def mark_query_started():

    global active_query_count

    active_query_count += 1

    try:

        await redis_client.incr(
            QUERY_ACTIVE_KEY
        )

        await redis_client.expire(
            QUERY_ACTIVE_KEY,
            QUERY_ACTIVE_TTL_SECONDS
        )

    except Exception as e:

        print(
            "\n[QUERY PAUSE FLAG ERROR] "
            "Could not mark query started:",
            repr(e)
        )


async def mark_query_finished():

    global active_query_count

    active_query_count = max(
        active_query_count - 1,
        0
    )

    try:

        remaining = await redis_client.decr(
            QUERY_ACTIVE_KEY
        )

        if remaining <= 0:

            await redis_client.delete(
                QUERY_ACTIVE_KEY
            )

    except Exception as e:

        print(
            "\n[QUERY PAUSE FLAG ERROR] "
            "Could not mark query finished:",
            repr(e)
        )


async def is_query_active():

    if active_query_count > 0:
        return True

    try:

        value = await redis_client.get(
            QUERY_ACTIVE_KEY
        )

        return int(value or 0) > 0

    except Exception as e:

        print(
            "\n[QUERY PAUSE FLAG ERROR] "
            "Could not read query flag:",
            repr(e)
        )

        return active_query_count > 0


async def wait_if_query_active(worker_name):

    paused = False

    while await is_query_active():

        if not paused:

            print(
                f"\n[{worker_name}] "
                "Query active; pausing background work"
            )

            paused = True

        await asyncio.sleep(1)

    if paused:

        print(
            f"\n[{worker_name}] "
            "Query finished; resuming background work"
        )


async def interruptible_background_sleep(seconds, worker_name):

    remaining = seconds

    while remaining > 0:

        await wait_if_query_active(worker_name)

        step = min(
            remaining,
            1
        )

        await asyncio.sleep(step)

        remaining -= step


def now_iso():

    return datetime.utcnow().isoformat()


def seven_days_ago_str():

    return (
        datetime.utcnow() - timedelta(days=7)
    ).strftime("%Y-%m-%d")


def generate_id(item):

    return hashlib.md5(
        (
            item["title"] + item["url"]
        ).encode()
    ).hexdigest()


def to_timestamp(dt_str):

    return int(
        datetime.fromisoformat(
            dt_str.replace("Z", "")
        ).timestamp()
    )


async def load_sliding_window_items():

    try:

        raw_items = await redis_client.zrevrange(
            ZSET_KEY,
            0,
            199
        )

    except Exception as e:

        print(
            "\n[AI NEWS GET] Redis fallback unavailable:",
            repr(e)
        )

        return []

    items = []

    for raw in raw_items:

        try:

            item = json.loads(raw)

            if isinstance(item, dict):
                items.append(item)

        except Exception as e:

            print(
                "\n[AI NEWS GET] Skipping malformed cached item:",
                repr(e)
            )

    return items


def group_items_by_source(items):

    grouped = {
        "github": [],
        "papers": [],
        "news": []
    }

    for item in items:

        source = item.get("source")

        if source in grouped:
            grouped[source].append(item)

    return grouped


async def fetch_github():

    headers = {
        "Authorization":
        f"Bearer {GITHUB_TOKEN}"
    }

    url = (
        "https://api.github.com/search/"
        "repositories?"
        f"q=created:>{seven_days_ago_str()}"
        "&sort=stars"
        "&order=desc"
        "&per_page=50"
    )

    async with httpx.AsyncClient(
        timeout=15
    ) as client:

        res = await client.get(
            url,
            headers=headers
        )

        data = res.json()

    items = []

    for repo in data.get("items", []):

        title = repo["full_name"].lower()

        if any(
            k in title
            for k in BLOCKED_KEYWORDS
        ):
            continue

        items.append({
            "title": repo["full_name"],
            "url": repo["html_url"],
            "source": "github",
            "created_at": repo["created_at"],
            "fetched_at": now_iso(),
            "meta": {
                "stars":
                repo["stargazers_count"],

                "language":
                repo.get("language")
            }
        })

    print(
        f"\n[GITHUB] "
        f"Fetched {len(items)} repos"
    )

    return items


async def fetch_papers():

    url = (
        "https://export.arxiv.org/api/query?"
        "search_query=cat:cs.AI"
        "&sortBy=submittedDate"
        "&max_results=25"
    )

    headers = {
        "User-Agent":
        "FusionAI/1.0"
    }

    async with httpx.AsyncClient(
        timeout=20
    ) as client:

        res = await client.get(
            url,
            headers=headers
        )

    print(
        f"\n[ARXIV STATUS] "
        f"{res.status_code}"
    )

    if res.status_code == 429:

        print(
            "\n[ARXIV ERROR RESPONSE] "
            "Rate exceeded."
        )

        return []

    if not res.text.strip():

        print(
            "\n[ARXIV] Empty response"
        )

        return []

    try:

        root = ET.fromstring(res.text)

    except Exception as e:

        print(
            "\n[ARXIV XML ERROR]",
            e
        )

        return []

    items = []

    for entry in root.findall(
        "{http://www.w3.org/2005/Atom}entry"
    ):

        try:

            published = entry.find(
                "{http://www.w3.org/2005/Atom}"
                "published"
            ).text

            pub_dt = datetime.fromisoformat(
                published.replace("Z", "")
            )

            if pub_dt < (
                datetime.utcnow()
                - timedelta(days=7)
            ):
                continue

            items.append({
                "title": entry.find(
                    "{http://www.w3.org/2005/Atom}"
                    "title"
                ).text.strip(),

                "url": entry.find(
                    "{http://www.w3.org/2005/Atom}"
                    "id"
                ).text,

                "source": "papers",

                "created_at": published,

                "fetched_at": now_iso(),

                "meta": {}
            })

        except Exception as e:

            print(
                "\n[ARXIV ENTRY ERROR]",
                e
            )

    print(
        f"\n[ARXIV] "
        f"Fetched {len(items)} papers"
    )

    return items


async def fetch_news():

    url = (
        "https://newsapi.org/v2/everything?"
        "q=artificial intelligence"
        f"&from={seven_days_ago_str()}"
        "&sortBy=publishedAt"
        "&pageSize=50"
        f"&apiKey={NEWS_API_KEY}"
    )

    async with httpx.AsyncClient(
        timeout=15
    ) as client:

        res = await client.get(url)

        data = res.json()

    items = []

    for article in data.get(
        "articles",
        []
    ):

        items.append({
            "title": article["title"],
            "url": article["url"],
            "source": "news",
            "created_at":
            article["publishedAt"],
            "fetched_at": now_iso(),
            "meta": {
                "source_name":
                article["source"]["name"]
            }
        })

    print(
        f"\n[NEWS] "
        f"Fetched {len(items)} articles"
    )

    return items


async def log_activity(source, count):

    key = f"{ACTIVITY_PREFIX}:{source}"

    now = int(
        datetime.utcnow().timestamp()
    )

    pipe = redis_client.pipeline()

    for _ in range(count):

        pipe.zadd(key, {str(now): now})

    pipe.zremrangebyscore(
        key,
        0,
        now - 3600
    )

    await pipe.execute()


async def get_activity_level(source):

    key = f"{ACTIVITY_PREFIX}:{source}"

    now = int(
        datetime.utcnow().timestamp()
    )

    return await redis_client.zcount(
        key,
        now - 3600,
        now
    )


def get_dynamic_interval(activity):

    if activity > 200:
        return 60

    elif activity > 100:
        return 180

    elif activity > 50:
        return 300

    return 600


async def allow_request(
    source,
    limit=20,
    window=60
):

    key = f"{RATE_PREFIX}:{source}"

    now = int(
        datetime.utcnow().timestamp()
    )

    pipe = redis_client.pipeline()

    pipe.zadd(key, {str(now): now})

    pipe.zremrangebyscore(
        key,
        0,
        now - window
    )

    pipe.zcount(
        key,
        now - window,
        now
    )

    _, _, count = await pipe.execute()

    return count <= limit


async def store_sliding_window(data):

    pipe = redis_client.pipeline()

    for item in data:

        item_id = generate_id(item)

        if not await redis_client.sismember(
            SEEN_KEY,
            item_id
        ):

            print(
                f"\n[EXTRACTION] "
                f"{item['title']}"
            )

            from backend.routes.Research_Routes.Extraction.extractor import (
                extract_full_document
            )

            full_text = await extract_full_document(
                item
            )

            if not full_text:

                print(
                    "\n[SKIPPED] "
                    "No content extracted"
                )

                continue

            await redis_client.sadd(
                SEEN_KEY,
                item_id
            )

            enriched_item = {
                **item,
                "content": full_text,
                "source_score": source_authority_score(item)
            }

            score = to_timestamp(
                item["created_at"]
            )

            pipe.zadd(
                ZSET_KEY,
                {
                    json.dumps(
                        enriched_item
                    ): score
                }
            )

    cutoff = int(
        (
            datetime.utcnow()
            - timedelta(days=7)
        ).timestamp()
    )

    pipe.zremrangebyscore(
        ZSET_KEY,
        0,
        cutoff
    )

    await pipe.execute()

    print(
        "\n[REDIS] Sliding window updated"
    )


async def adaptive_fetch(
    source,
    fetch_func
):

    try:

        await wait_if_query_active(
            f"SCHEDULER {source.upper()}"
        )

        activity = await get_activity_level(
            source
        )

        interval = get_dynamic_interval(
            activity
        )

        if not await allow_request(source):

            print(
                f"\n[RATE LIMITED] "
                f"{source}"
            )

            return [], interval

        await wait_if_query_active(
            f"SCHEDULER {source.upper()}"
        )

        data = await fetch_func()

        await wait_if_query_active(
            f"SCHEDULER {source.upper()}"
        )

        await log_activity(
            source,
            len(data)
        )

    except Exception as e:

        print(
            f"\n[FETCH ERROR] "
            f"{source}: {e}"
        )

        return [], 600

    return data, interval


async def adaptive_scheduler():

    global latest_live_data

    while True:

        try:

            await wait_if_query_active(
                "ADAPTIVE SCHEDULER"
            )

            print(
                "\n=============================="
            )

            print(
                "\nStarting adaptive "
                "fetch cycle..."
            )

            g_data, g_int = await adaptive_fetch(
                "github",
                fetch_github
            )

            await interruptible_background_sleep(
                3,
                "ADAPTIVE SCHEDULER"
            )

            p_data, p_int = await adaptive_fetch(
                "papers",
                fetch_papers
            )

            await interruptible_background_sleep(
                3,
                "ADAPTIVE SCHEDULER"
            )

            n_data, n_int = await adaptive_fetch(
                "news",
                fetch_news
            )

            await wait_if_query_active(
                "ADAPTIVE SCHEDULER"
            )

            latest_live_data = {
                "github": g_data,
                "papers": p_data,
                "news": n_data
            }

            print(
                "\n[LIVE CACHE UPDATED]"
            )

            await store_sliding_window(
                g_data + p_data + n_data
            )

            next_interval = min(
                g_int,
                p_int,
                n_int
            )

            print(
                f"\nNext fetch cycle "
                f"in {next_interval}s"
            )

            await interruptible_background_sleep(
                next_interval,
                "ADAPTIVE SCHEDULER"
            )

        except Exception as e:

            print(
                "\n[SCHEDULER ERROR]",
                e
            )

            await interruptible_background_sleep(
                60,
                "ADAPTIVE SCHEDULER"
            )


async def background_ingestion_worker():

    while True:

        try:

            await wait_if_query_active(
                "INGESTION WORKER"
            )

            print(
                "\n[INGESTION WORKER] "
                "Starting processing cycle"
            )

            from backend.routes.Research_Routes.processor import (
                process_and_build_dataset
            )

            await process_and_build_dataset(
                redis_client,
                pause_callback=lambda: wait_if_query_active(
                    "INGESTION PROCESSOR"
                )
            )

            await wait_if_query_active(
                "INGESTION WORKER"
            )

            print(
                "\n[INGESTION WORKER] "
                "Cycle completed"
            )

            await interruptible_background_sleep(
                10,
                "INGESTION WORKER"
            )

        except Exception as e:

            print(
                "\n[INGESTION WORKER ERROR]",
                e
            )

            await interruptible_background_sleep(
                5,
                "INGESTION WORKER"
            )


@router.on_event("startup")
async def start_background_services():

    print(
        "\nStarting FusionAI services..."
    )

    asyncio.create_task(
        adaptive_scheduler()
    )

    asyncio.create_task(
        background_ingestion_worker()
    )

    print(
        "\nFusionAI services started"
    )


@router.post("/query")
async def submit_query(payload: QueryRequest):

    query = payload.query

    if not query:

        return {
            "success": False,
            "message": "Query missing"
        }

    try:

        await mark_query_started()

        print(
            f"\n[QUERY RECEIVED] "
            f"{query}"
        )

        from backend.routes.Research_Routes.Query_Pipeline.query_controller import (
            query_controller
        )

        print(
            "\n[QUERY CONTROLLER] Starting process"
        )

        response = await query_controller.process(
            query,
            mode=payload.mode
        )

        print(
            "\n[QUERY COMPLETED]"
        )

        evaluation = None

        try:

            from backend.routes.Research_Routes.benchmark_service import (
                evaluate_query_response
            )

            evaluation = await evaluate_query_response(
                query,
                response
            )

        except Exception as eval_error:

            print(
                "\n[QUERY EVALUATION ERROR]",
                repr(eval_error)
            )

            evaluation = {
                "error": repr(eval_error)
            }

        return {
            "success": True,
            "query": query,
            "response": response,
            "evaluation": evaluation
        }

    except Exception as e:

        print(
            "\n[QUERY ERROR]",
            repr(e)
        )

        return {
            "success": False,
            "message": str(e)
        }

    finally:

        await mark_query_finished()

@router.get("/get")
async def get_ai_pulse():

    live_data = (
        latest_live_data["github"]
        + latest_live_data["papers"]
        + latest_live_data["news"]
    )

    combined = (
        live_data
    )

    source_counts = {
        "github": len(latest_live_data["github"]),
        "papers": len(latest_live_data["papers"]),
        "news": len(latest_live_data["news"])
    }

    if not combined:

        cached_items = await load_sliding_window_items()
        cached_grouped = group_items_by_source(
            cached_items
        )

        if cached_items:

            combined = cached_items
            source_counts = {
                "github": len(cached_grouped["github"]),
                "papers": len(cached_grouped["papers"]),
                "news": len(cached_grouped["news"])
            }

            print(
                "\n[AI NEWS GET] Served from Redis sliding window: "
                f"{len(combined)} items"
            )

        else:

            print(
                "\n[AI NEWS GET] No in-memory or Redis live data available"
            )

    return {
        "data": combined,
        "live_counts": {
            "github":
            source_counts["github"],

            "papers":
            source_counts["papers"],

            "news":
            source_counts["news"]
        },
        "last_updated": now_iso()
    }
