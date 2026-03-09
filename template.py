import os
from pathlib import Path 

list_of_filepaths = [
    "frontend/",
    "backend/",
    "backend/core/",
    "backend/core/logging.py",
    "backend/core/exceptions.py",
    "services/",
    "services/__init__.py",

    "services/research_service/",
    "services/research_service/__init__.py",

    "services/research_service/data_processing/",
    "services/research_service/data_processing/__init__.py",

    "services/research_service/data_processing/doc_processing/",
    "services/research_service/data_processing/doc_processing/__init__.py",
    "services/research_service/data_processing/doc_processing/doc_processor.py",

    "services/research_service/data_processing/audio_processing/",
    "services/research_service/data_processing/audio_processing/__init__.py",
    "services/research_service/data_processing/audio_processing/audio_transcriber.py",
    "services/research_service/data_processing/audio_processing/youtube_transcriber.py",

    "services/research_service/data_processing/web_scraping/",
    "services/research_service/data_processing/web_scraping/__init__.py",
    "services/research_service/data_processing/web_scraping/web_scraper.py",

    "services/research_service/embeddings/",
    "services/research_service/embeddings/__init__.py",
    "services/research_service/embeddings/embedding_generator.py",

    "services/research_service/vector_database/",
    "services/research_service/vector_database/__init__.py",
    "services/research_service/vector_database/vector_database.py",

    "services/research_service/podcast/",
    "services/research_service/podcast/__init__.py",
    "services/research_service/podcast/script_generator.py",
    "services/research_service/podcast/text_to_speech.py"
]

for filepath in list_of_filepaths:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filepath.suffix == "":
        os.makedirs(filepath, exist_ok=True)
    else:
        os.makedirs(filedir, exist_ok=True)
        if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
            with open(filepath, "w") as f:
                pass
        else:
            print(f"File already exists: {filepath}")
