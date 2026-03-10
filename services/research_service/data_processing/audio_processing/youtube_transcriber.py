import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import assemblyai as aai
from posthog import api_key
from pydot import List, Optional
import yt_dlp
from services.research_service.data_processing.doc_processing.doc_processor import DocumentChunk
load_dotenv()

aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

transcriber=aai.Transcriber()

class YoutubeTranscriber:
    def __init__(self):
        self.transcriber=aai.Transcriber()
        self.temp_dir = Path(tempfile.gettempdir()) / "youtube_transcriber"
        self.temp_dir.mkdir(exist_ok=True)
    
    def extract_video_id(self, url: str) -> Optional[str]:
        if "v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1].split("?")[0]
        else:
            video_id = None
        return video_id
    
    def download_audio(self, url: str) -> str:
        video_id = self.extract_video_id(url)
        if not video_id:
            raise ValueError("Could not extract video ID from URL")
        
        expected_path = self.temp_dir / f"{video_id}.m4a"
        if expected_path.exists():
            print(f"Audio already exists: {expected_path}")
            return str(expected_path)
        
        ydl_opts = {
            'format': 'm4a/bestaudio/best',
            'outtmpl': str(self.temp_dir / '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download([url])
            
            if error_code != 0:
                raise Exception(f"yt-dlp download failed with error code: {error_code}")
        
        if not expected_path.exists():
            raise FileNotFoundError(f"Expected audio file not found: {expected_path}")
        

        return str(expected_path)

    def transcribe_youtube_video(
        self,
        url: str,
        cleanup_audio: bool = True
    ) -> List[DocumentChunk]:
        try:
            audio_path = self.download_audio(url)
            
            config = aai.TranscriptionConfig(
                speech_models=[aai.SpeechModel.universal],
                speaker_labels=True,
                punctuate=True
            )
            transcriber = aai.Transcriber(config=config)
            transcript = transcriber.transcribe(audio_path)
            
            if transcript.status == aai.TranscriptStatus.error:
                raise Exception(f"Transcription failed: {transcript.error}")
            
            chunks = []
            video_id = self.extract_video_id(url)
            for i, utterance in enumerate(transcript.utterances):
                chunk = DocumentChunk(
                    content=f"Speaker {utterance.speaker}: {utterance.text}",
                    source_file=f"YouTube Video {video_id}",
                    source_type="youtube",
                    page_number=None,
                    chunk_index=i,
                    start_char=utterance.start,
                    end_char=utterance.end,
                    metadata={
                        'speaker': utterance.speaker,
                        'start_time': utterance.start,
                        'end_time': utterance.end,
                        'confidence': getattr(utterance, 'confidence', None),
                        'video_url': url,
                        'video_id': video_id
                    }
                )
                chunks.append(chunk)
            
            
            return chunks
        except Exception as e:
            print(f"Error processing Youtube Videos:{e}")
            raise e

if __name__ == "__main__":
    
    transcriber = YoutubeTranscriber()
    
    try:
        test_url = "https://www.youtube.com/watch?v=D26sUZ6DHNQ"
        chunks = transcriber.transcribe_youtube_video(test_url)
        
        print(f"Transcribed {len(chunks)} utterances:")
        for chunk in chunks[:5]:
            print(f"  {chunk.content}")
        
    except Exception as e:
        print(f"Error: {e}")