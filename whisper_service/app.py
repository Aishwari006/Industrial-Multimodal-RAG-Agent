from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from whisper_model import transcribe_audio

# ---------------------------- #

app = FastAPI(title="Whisper Speech-to-Text API")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ---------------------------- #

@app.get("/")
def health():
    return {
        "status": "running",
        "service": "Whisper Speech-to-Text"
    }

# ---------------------------- #

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):

    # Validate file type
    if not file.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=400,
            detail="Only audio files are supported."
        )

    # Generate unique filename
    extension = Path(file.filename).suffix

    temp_filename = f"{uuid.uuid4()}{extension}"

    temp_path = UPLOAD_DIR / temp_filename

    try:

        # Save uploaded audio

        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run Whisper

        transcript = transcribe_audio(str(temp_path))

        return JSONResponse(
            {
                "success": True,
                "transcript": transcript
            }
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        if temp_path.exists():
            temp_path.unlink()