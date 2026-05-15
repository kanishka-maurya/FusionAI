from backend.routes.Research_Routes.chunker import chunk_text
from backend.routes.Research_Routes.dataset_builder import save_sample
from backend.routes.Research_Routes.models.llm_loader import summarize_with_gemini

from backend.routes.Research_Routes.Nexus_Graph_DB.services import engine_services

import json
import hashlib
import asyncio


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


def generate_id(item):

    return hashlib.md5(
        (
            item["title"] +
            item["url"]
        ).encode()
    ).hexdigest()


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

            wait_time = min(
                2 ** attempt,
                30
            )

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

            already_processed = await redis_client.sismember(
                PROCESSED_KEY,
                doc_id
            )

            if already_processed:

                print(
                    f"Already processed: "
                    f"{item['title']}"
                )

                continue

            title_lower = item.get(
                "title",
                ""
            ).lower()

            if any(
                word in title_lower
                for word in BLOCKED_WORDS
            ):

                print(
                    f"Skipping suspicious repo: "
                    f"{item['title']}"
                )

                continue

            content = item.get("content")

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

            success_count = 0

            for idx, chunk in enumerate(chunks):

                try:

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

                        print(
                            "Invalid structured output"
                        )

                        continue

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

                    if not isinstance(key_points, list):
                        key_points = []

                    if not isinstance(entities, list):
                        entities = []

                    key_points = [
                        str(kp).strip()
                        for kp in key_points
                        if str(kp).strip()
                    ]

                    entities = [
                        str(ent).strip()
                        for ent in entities
                        if str(ent).strip()
                    ]

                    save_sample(
                        chunk,
                        summary_json
                    )

                    print(
                        "Dataset sample saved"
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

                        success_count += 1

                    except Exception as graph_error:

                        print(
                            f"Graph insertion failed: "
                            f"{graph_error}"
                        )

                    await asyncio.sleep(1)

                except Exception as chunk_error:

                    print(
                        f"Chunk processing failed: "
                        f"{chunk_error}"
                    )

            if success_count > 0:

                await redis_client.sadd(
                    PROCESSED_KEY,
                    doc_id
                )

                print(
                    f"\nFinished document: "
                    f"{item['title']}"
                )

                print(
                    f"Successfully processed "
                    f"{success_count}/{len(chunks)} chunks"
                )

            else:

                print(
                    f"\nNo successful chunks for: "
                    f"{item['title']}"
                )

        except Exception as e:

            print(
                f"\nProcessor error: {e}"
            )