from fastapi import APIRouter
from datetime import datetime, timedelta

import httpx
import asyncio
import json
import hashlib
import xml.etree.ElementTree as ET
import redis.asyncio as redis
import os

from backend.routes.Research_Routes.Extraction.extractor import (
    extract_full_document
)

from backend.routes.Research_Routes.processor import (
    process_and_build_dataset
)

from backend.routes.Research_Routes.Query_Pipeline.query_controller import (
    query_controller
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

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

latest_live_data = {
    "github": [],
    "papers": [],
    "news": []
}


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
                "content": full_text
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

        data = await fetch_func()

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

            await asyncio.sleep(3)

            p_data, p_int = await adaptive_fetch(
                "papers",
                fetch_papers
            )

            await asyncio.sleep(3)

            n_data, n_int = await adaptive_fetch(
                "news",
                fetch_news
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

            await asyncio.sleep(
                next_interval
            )

        except Exception as e:

            print(
                "\n[SCHEDULER ERROR]",
                e
            )

            await asyncio.sleep(60)


async def background_ingestion_worker():

    while True:

        try:

            print(
                "\n[INGESTION WORKER] "
                "Starting processing cycle"
            )

            await process_and_build_dataset(
                redis_client
            )

            print(
                "\n[INGESTION WORKER] "
                "Cycle completed"
            )

            await asyncio.sleep(10)

        except Exception as e:

            print(
                "\n[INGESTION WORKER ERROR]",
                e
            )

            await asyncio.sleep(5)


#@router.on_event("startup")
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
async def submit_query(payload: dict):

    query = payload.get("query")

    if not query:

        return {
            "success": False,
            "message": "Query missing"
        }

    try:

        print(
            f"\n[QUERY RECEIVED] "
            f"{query}"
        )

        response = await query_controller.process(
            query
        )

        print(
            "\n[QUERY COMPLETED]"
        )

        return {
            "success": True,
            "query": query,
            "response": response
        }

    except Exception as e:

        print(
            "\n[QUERY ERROR]",
            e
        )

        return {
            "success": False,
            "message": str(e)
        }


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

    return {
        "data": combined,
        "live_counts": {
            "github":
            len(latest_live_data["github"]),

            "papers":
            len(latest_live_data["papers"]),

            "news":
            len(latest_live_data["news"])
        },
        "last_updated": now_iso()
    }