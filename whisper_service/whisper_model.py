from faster_whisper import WhisperModel

# ---------------- Configuration ---------------- #

MODEL_NAME = "small.en"

DEVICE = "cpu"
# Change to "cuda" if using NVIDIA GPU.

COMPUTE_TYPE = "int8"
# CPU:
# int8 -> fastest
#
# GPU:
# float16

# ---------------- Load Model ---------------- #

print("Loading Whisper model...")

model = WhisperModel(
    MODEL_NAME,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
)

print("Whisper model loaded successfully.")


# ---------------- Transcription Function ---------------- #

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes an English audio file and returns the transcript.
    """

    segments, info = model.transcribe(
        audio_path,
        language="en",          # Force English
        beam_size=1,
        vad_filter=True,        # Remove silence
    )

    transcript = []

    for segment in segments:
        transcript.append(segment.text)

    return " ".join(transcript).strip()