from functools import lru_cache


@lru_cache(maxsize=1)
def get_embedding_generator():
    from services.research_service.embeddings.embedding_generator import EmbeddingGenerator

    return EmbeddingGenerator()


@lru_cache(maxsize=1)
def get_vector_db():
    from services.research_service.vector_database.vector_database import ChromaVectorDatabase

    return ChromaVectorDatabase()
