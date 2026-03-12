import os
import time
from tracemalloc import start
from dotenv import load_dotenv
import assemblyai as aai
from backend.core.logging import logging
from services.research_service.data_processing.doc_processing.doc_processor import DocumentChunk

load_dotenv()

aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

transcriber=aai.Transcriber()

def transcribe_audio(file_path):
    transcript=transcriber.transcribe(file_path)
    if transcript.status == aai.TranscriptStatus.error:
       return f"Error:{transcript.error}"
    return transcript.text

class AudioTranscriber:
    def __init__(self):
        self.transcriber = aai.Transcriber()

    def _ms_to_timestamp(self, ms):
        """Helper to convert milliseconds to [MM:SS] format."""
        seconds = int((ms / 1000) % 60)
        minutes = int((ms / (1000 * 60)) % 60)
        return f"{minutes:02d}:{seconds:02d}"

    
    def _create_chunks_by_chapters(self, transcript, source_file="audio_source"):
        chunks = []
        start = time.time()

        for idx, chapter in enumerate(transcript.chapters):

           chapter_utterances = [
            f"[{self._ms_to_timestamp(u.start)}] Speaker {u.speaker}: {u.text}"
            for u in transcript.utterances
            if chapter.start <= u.start <= chapter.end
           ]

           chapter_text = "\n".join(chapter_utterances)

           metadata = {
            "headline": chapter.headline,
            "start_time_ms": chapter.start,
            "end_time_ms": chapter.end,
            "time_range": f"{self._ms_to_timestamp(chapter.start)} - {self._ms_to_timestamp(chapter.end)}",
            "source": "audio_transcript"
           }

           chunk = DocumentChunk(
            content=chapter_text,
            source_file=source_file,
            source_type="audio",
            page_number=None,
            chunk_index=idx,
            start_char=0,
            end_char=len(chapter_text),
            metadata=metadata
           )

           chunks.append(chunk)

           end = time.time()
           logging.info(f"Chapter chunking time: {end - start:.2f} seconds")

           return chunks
    
    def run_notebook_pipeline(self, audio_url):
        transcript = self.transcriber.transcribe(
            audio_url, 
            config=aai.TranscriptionConfig(speech_models=[aai.SpeechModel.universal], speaker_labels=True, auto_chapters=True)
        )
        
        structured_chunks = self._create_chunks_by_chapters(transcript, source_file=audio_url)
    
        
        return {
          "chunks": structured_chunks,
          "full_text": transcript.text
        }
    
if __name__ == "__main__":
    #sample audio file with clear chapters and speaker labels for testing
    audio_url = "https://storage.googleapis.com/aai-web-samples/5_common_sports_injuries.mp3"

    pipeline = AudioTranscriber()

    result = pipeline.run_notebook_pipeline(audio_url)

    print("\nFULL TRANSCRIPT\n")
    print(result["full_text"])

    print("\nCHUNKS\n")

    for chunk in result["chunks"]:
       print("\n------------------------")
       print(f"Chunk ID: {chunk.chunk_id}")
       print(f"Headline: {chunk.metadata.get('headline')}")
       print(f"Time: {chunk.metadata.get('time_range')}")
       print(chunk.content)