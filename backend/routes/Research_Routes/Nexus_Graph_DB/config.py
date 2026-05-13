from services.research_service.embeddings.embedding_generator import EmbeddingGenerator

embedding_service = EmbeddingGenerator(
    model_name="gemini-embedding-001"
)

sample_vector = embedding_service.generate_query_embedding("test")

class Settings:
    EMBEDDING_DIM : int= len(sample_vector)
    MODEL_NAME: str = "gemini-embedding-001"

settings = Settings()

embedding_service = EmbeddingGenerator(
    model_name=settings.MODEL_NAME
)