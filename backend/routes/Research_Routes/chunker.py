from backend.routes.Research_Routes.utils import semantic_chunk_text


def chunk_text(text, chunk_size=1200, overlap=180):
    return semantic_chunk_text(
        text,
        chunk_size=chunk_size,
        overlap=overlap
    )
