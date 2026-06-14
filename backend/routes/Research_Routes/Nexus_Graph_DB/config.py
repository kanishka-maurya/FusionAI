class Settings:
    EMBEDDING_DIM: int = 768
    MODEL_NAME: str = "gemini-embedding-001"

settings = Settings()

class LazyEmbeddingService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._service = None

    def _get_service(self):
        if self._service is None:
            from services.research_service.embeddings.embedding_generator import (
                EmbeddingGenerator,
            )

            self._service = EmbeddingGenerator(
                model_name=self.model_name
            )
        return self._service

    def generate_query_embedding(self, query_text: str):
        return self._get_service().generate_query_embedding(
            query_text
        )


embedding_service = LazyEmbeddingService(
    model_name=settings.MODEL_NAME
)
