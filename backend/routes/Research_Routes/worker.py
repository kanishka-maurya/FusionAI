import asyncio
import json
import redis.asyncio as redis

from backend.routes.Research_Routes.processor import (
    process_and_build_dataset
)

from backend.routes.Research_Routes.Query_Pipeline.query_controller import (
    query_controller
)

from backend.routes.Research_Routes.get_Latest_Data import (
    adaptive_scheduler,
    interruptible_background_sleep,
    mark_query_finished,
    mark_query_started,
    wait_if_query_active,
)
redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)

QUERY_QUEUE = "ai:query_queue"


async def background_ingestion_worker():

    while True:

        try:

            await wait_if_query_active(
                "STANDALONE INGESTION WORKER"
            )

            print(
                "\n[INGESTION WORKER] "
                "Starting processing cycle"
            )

            await process_and_build_dataset(
                redis_client,
                pause_callback=lambda: wait_if_query_active(
                    "STANDALONE INGESTION PROCESSOR"
                )
            )

            await wait_if_query_active(
                "STANDALONE INGESTION WORKER"
            )

            print(
                "\n[INGESTION WORKER] "
                "Cycle completed"
            )

            await interruptible_background_sleep(
                10,
                "STANDALONE INGESTION WORKER"
            )

        except Exception as e:

            print(
                "\n[INGESTION WORKER ERROR]",
                e
            )

            await interruptible_background_sleep(
                5,
                "STANDALONE INGESTION WORKER"
            )


async def query_worker():

    while True:

        try:

            result = await redis_client.blpop(
                QUERY_QUEUE
            )

            if not result:
                continue

            _, raw = result

            payload = json.loads(raw)

            query = payload["query"]

            print(
                f"\n[QUERY WORKER] "
                f"Received query: {query}"
            )

            await mark_query_started()

            try:

                response = await query_controller.process(
                    query,
                    mode=payload.get(
                        "mode",
                        "deep_research"
                    )
                )

            finally:

                await mark_query_finished()

            print(
                "\n[QUERY WORKER] "
                "Pipeline completed"
            )

            print(response)

        except Exception as e:

            print(
                "\n[QUERY WORKER ERROR]",
                e
            )

            await asyncio.sleep(5)


async def main():

    await asyncio.gather(
        adaptive_scheduler(),
        background_ingestion_worker(),
        #query_worker()
    )


if __name__ == "__main__":

    asyncio.run(main())
