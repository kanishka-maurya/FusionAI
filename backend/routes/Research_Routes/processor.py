from backend.routes.Research_Routes.chunker import chunk_text
from backend.routes.Research_Routes.dataset_builder import save_sample
from backend.routes.Research_Routes.models.llm_loader import summarize_with_gemini

import json
import hashlib
import asyncio

PROCESSED_KEY = "ai:processed_docs"

semaphore = asyncio.Semaphore(3)


def generate_id(item):
    return hashlib.md5(
        (item["title"] + item["url"]).encode()
    ).hexdigest()


async def summarize_with_retry(chunk, retries=5):
    """
    Gemini retry with exponential backoff
    """

    async with semaphore:

        for attempt in range(retries):

            try:
                summary = await summarize_with_gemini(chunk)

                if summary:
                    return summary

            except Exception as e:
                print(f"Gemini error: {e}")

            wait_time = 2 ** attempt

            print(
                f"Retrying in {wait_time}s "
                f"(attempt {attempt + 1}/{retries})"
            )

            await asyncio.sleep(wait_time)

    return None


async def process_and_build_dataset(redis_client):

    data = await redis_client.zrevrange(
        "ai:raw:current",
        0,
        10
    )

    print(f"\nProcessing {len(data)} documents...\n")

    for raw in data:

        try:
            item = json.loads(raw)

            doc_id = generate_id(item)

            if await redis_client.sismember(PROCESSED_KEY, doc_id):
                print(f"Already processed: {item['title']}")
                continue

            content = item.get("content")

            if not content:
                print(f"No content found: {item['title']}")
                continue

            print("\n" + "=" * 80)
            print(f"DOCUMENT: {item['title']}")
            print("=" * 80)

            chunks = chunk_text(content)

            print(f"Total chunks: {len(chunks)}")

            for idx, chunk in enumerate(chunks):

                print(f"\nProcessing chunk {idx + 1}/{len(chunks)}")

                summary = await summarize_with_retry(chunk)

                if not summary:
                    print("Failed summary generation")
                    continue

                save_sample(chunk, summary)

                print("Summary saved successfully")

                await asyncio.sleep(1)

            await redis_client.sadd(PROCESSED_KEY, doc_id)

            print(f"\nFinished document: {item['title']}")

        except Exception as e:

            print(f"\nProcessor error: {e}")