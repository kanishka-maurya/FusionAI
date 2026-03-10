import os
from dotenv import load_dotenv
import assemblyai as aai

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
    
    def process(self, audio_source: str, use_speakers: bool = False):
        """Transcribes audio with optional speaker diarization."""
        config = aai.TranscriptionConfig(
            speech_models=[aai.SpeechModel.universal],
            speaker_labels=use_speakers,
            auto_highlights=True, 
            sentiment_analysis=True, 
            entity_detection=True   
        )  
        transcript = self.transcriber.transcribe(audio_source, config=config)
        
        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(f"Transcription failed: {transcript.error}")
            
        return transcript
    
    def _create_chunks_by_chapters(self, transcript):
        chapters_data = []
        
        for chapter in transcript.chapters:
            chapter_utterances = [
                f"[{self._ms_to_timestamp(u.start)}] Speaker {u.speaker}: {u.text}"
                for u in transcript.utterances
                if chapter.start <= u.start <= chapter.end
            ]
            
            chapters_data.append({
                "headline": chapter.headline,
                "time_range": f"{self._ms_to_timestamp(chapter.start)} - {self._ms_to_timestamp(chapter.end)}",
                "content": "\n".join(chapter_utterances)
            })
            
        return chapters_data
    
    def generate_notebook_insights(self, transcript):
        
        prompt = """
        Review the provided transcript and act as a precise Research Assistant.
        
        GOAL: Create a 'Source Guide' for this audio.
        
        REQUIRED SECTIONS:

        1. KEY DECISIONS: List every major decision. For EACH decision, you MUST 
           provide the exact [MM:SS] timestamp and the Speaker who made it.
           Example: 'Budget approved for Q3 [04:12] - Speaker A'
           
        2. THEMES & CONCEPTS: Identify the 3-5 main themes. For each, provide a 2-sentence explanation.

        3. CLAIM TRACKER: List the major claims or facts stated. Attribute each to a Speaker.

        4. Q&A GENERATOR: Suggest 5 questions a user might ask this 'Notebook' based on this audio.

        5. ACTION ITEMS: List tasks, the owner (Speaker), and the timestamp where it was assigned.
        
        6. SOURCE SUMMARY: A 3-sentence summary of the 'Knowledge' contained in this file.

        7. GLOSSARY: Define any technical terms or project codenames mentioned.

        Format everything in clean Markdown.

        """
        
        result = aai.Lemur().task(
            prompt=prompt,
            context=transcript.text,
            final_model=aai.LemurModel.claude3_5_sonnet
        )
        return result.response

    def run_notebook_pipeline(self, audio_url):
        transcript = self.transcriber.transcribe(
            audio_url, 
            config=aai.TranscriptionConfig(speech_models=[aai.SpeechModel.universal], speaker_labels=True, auto_chapters=True)
        )
        
        structured_source = self._create_chunks_by_chapters(transcript)
    
        
        return {  
            "ui_chapters": structured_source, 
            "full_text": transcript.text
        }
    
if __name__ == "__main__":
    #sample audio file with clear chapters and speaker labels for testing
    audio_url = "https://storage.googleapis.com/aai-web-samples/5_common_sports_injuries.mp3"

    pipeline = AudioTranscriber()

    result = pipeline.run_notebook_pipeline(audio_url)

    print("\nFULL TRANSCRIPT\n")
    print(result["full_text"])

    print("\nCHAPTERS\n")
    for chapter in result["ui_chapters"]:
        print(f"\n{chapter['headline']}")
        print(f"Time: {chapter['time_range']}")
        print(chapter["content"])