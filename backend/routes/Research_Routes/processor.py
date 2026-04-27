from backend.routes.Research_Routes.chunker import chunk_text
from backend.routes.Research_Routes.summarizer import summarize_chunk
from backend.routes.Research_Routes.dataset_builder import save_sample
from backend.routes.Research_Routes.models.llm_loader import load_quantized_model
import json
from backend.routes.Research_Routes.Extraction.extractor import extract_full_document
from backend.routes.Research_Routes.get_Latest_Data import generate_id

PROCESSED_KEY="ai:processed_docs"
tokenizer, model = load_quantized_model()
async def process_and_build_dataset(redis_client):
    data = await redis_client.zrevrange("ai:raw:current", 0, 10)

    for raw in data:
        item = json.loads(raw)
        doc_id = generate_id(item)

        if await redis_client.sismember(PROCESSED_KEY, doc_id):
            continue

        content = item.get("content")

        if not content:
            content = await extract_full_document(item)

        if not content:
            continue

        chunks = chunk_text(content)
        print("Currently processing...",chunks)
        for chunk in chunks:
            summary = summarize_chunk(chunk, tokenizer, model)
            save_sample(chunk, summary)

        await redis_client.sadd(PROCESSED_KEY, doc_id)