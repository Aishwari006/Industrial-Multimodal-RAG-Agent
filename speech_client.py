import requests
from dotenv import load_dotenv
import os
load_dotenv()

WHISPER_API_URL = os.getenv("WHISPER_API")

def transcribe_audio(audio_path: str) -> str:
    """
    Sends an audio file to the Whisper service
    and returns the transcript.
    """
    try:
        with open(audio_path, "rb") as audio_file:
            # FIX: Explicitly provide a filename and the 'audio/wav' content type
            files = {
                "file": ("audio.wav", audio_file, "audio/wav")
            }

            response = requests.post(
                WHISPER_API_URL,
                files=files,
                timeout=120,
            )

        response.raise_for_status()
        data = response.json()
        return data["transcript"]

    except requests.exceptions.ConnectionError:
        raise Exception(
            "Cannot connect to Whisper Service.\n"
            "Did you start FastAPI?"
        )

    except requests.exceptions.Timeout:
        raise Exception(
            "Whisper transcription timed out."
        )

    except Exception as e:
        raise Exception(str(e))
    