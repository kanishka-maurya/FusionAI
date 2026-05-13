from backend.routes.Research_Routes.chunker import chunk_text
from backend.routes.Research_Routes.dataset_builder import save_sample
from backend.routes.Research_Routes.models.llm_loader import summarize_with_gemini

from backend.routes.Research_Routes.Nexus_Graph_DB.services import engine_services
from backend.routes.Research_Routes.Nexus_Graph_DB.database import db

import json
import hashlib
import asyncio
import os
BLOCKED_WORDS = [
    "cheat",
    "hack",
    "exploit",
    "crack",
    "malware",
    "stealer",
    "token logger"
]
PROCESSED_KEY = "ai:processed_docs"

semaphore = asyncio.Semaphore(3)

GRAPH_DB_PATH = "storage/graph_db.json"


def generate_id(item):
    return hashlib.md5(
        (item["title"] + item["url"]).encode()
    ).hexdigest()

def persist_graph():

    os.makedirs("storage", exist_ok=True)

    payload = {
        "nodes": db.nodes,

        "inverted_index": {
            k: list(v)
            for k, v in db.inverted_index.items()
        },

        "global_active_roots": list(
            db.global_active_roots
        ),

        "node_counter": db.node_counter
    }

    with open(
        GRAPH_DB_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            payload,
            f,
            indent=2
        )


def load_graph():

    if not os.path.exists(GRAPH_DB_PATH):
        return

    with open(
        GRAPH_DB_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        payload = json.load(f)

    db.nodes = payload.get(
        "nodes",
        {}
    )

    db.inverted_index = {
        k: set(v)
        for k, v in payload.get(
            "inverted_index",
            {}
        ).items()
    }

    db.global_active_roots = set(
        payload.get(
            "global_active_roots",
            []
        )
    )

    db.node_counter = payload.get(
        "node_counter",
        0
    )

    print("Graph database loaded successfully")


load_graph()

async def summarize_with_retry(
    chunk,
    retries=5
):

    async with semaphore:

        for attempt in range(retries):

            try:

                summary = await summarize_with_gemini(
                    chunk
                )

                if summary:
                    return summary

            except Exception as e:

                print(
                    f"Gemini error: {e}"
                )

            wait_time = 2 ** attempt

            print(
                f"Retrying in {wait_time}s "
                f"(attempt {attempt + 1}/{retries})"
            )

            await asyncio.sleep(wait_time)

    return None


async def process_and_build_dataset(
    redis_client
):

    data = await redis_client.zrevrange(
        "ai:raw:current",
        0,
        10
    )

    print(
        f"\nProcessing {len(data)} documents...\n"
    )

    for raw in data:

        try:

            item = json.loads(raw)

            doc_id = generate_id(item)
            if await redis_client.sismember(
                PROCESSED_KEY,
                doc_id
            ):

                print(
                    f"Already processed: "
                    f"{item['title']}"
                )

                continue

            content = item.get("content")
            title_lower = item["title"].lower()

            if any(word in title_lower for word in BLOCKED_WORDS):
              print(f"Skipping suspicious repo: {item['title']}")
              continue
            if not content:

                print(
                    f"No content found: "
                    f"{item['title']}"
                )

                continue

            print("\n" + "=" * 80)

            print(
                f"DOCUMENT: "
                f"{item['title']}"
            )

            print("=" * 80)

            chunks = chunk_text(content)

            print(
                f"Total chunks: "
                f"{len(chunks)}"
            )

            for idx, chunk in enumerate(chunks):

                print(
                    f"\nProcessing chunk "
                    f"{idx + 1}/{len(chunks)}"
                )

                summary_json = await summarize_with_retry(
                    chunk
                )

                if not summary_json:

                    print(
                        "Failed summary generation"
                    )

                    continue
                required_fields = [
                  "summary",
                  "key_points",
                  "entities"
                ]

                if not all(
                    field in summary_json
                    for field in required_fields
                ):
                   print("Invalid structured output")
                   continue
                save_sample(
                    chunk,
                    summary_json
                )

                print(
                    "Dataset sample saved"
                )

                summary = summary_json.get(
                    "summary",
                    ""
                )

                key_points = summary_json.get(
                    "key_points",
                    []
                )

                entities = summary_json.get(
                    "entities",
                    []
                )

                try:

                    node_id = await engine_services.ingest_document(
                        chunk_id=f"{doc_id}_{idx}",
                        content=chunk,
                        summary=summary,
                        key_points=key_points,
                        entities=entities
                    )

                    print(
                        f"Inserted node into graph: "
                        f"{node_id}"
                    )

                    persist_graph()

                    print(
                        "Graph persisted successfully"
                    )

                except Exception as graph_error:

                    print(
                        f"Graph insertion failed: "
                        f"{graph_error}"
                    )

                await asyncio.sleep(1)
            await redis_client.sadd(
                PROCESSED_KEY,
                doc_id
            )

            print(
                f"\nFinished document: "
                f"{item['title']}"
            )

        except Exception as e:

            print(
                f"\nProcessor error: {e}"
            )