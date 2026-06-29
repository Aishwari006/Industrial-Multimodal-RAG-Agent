from whisper_model import transcribe_audio

text = transcribe_audio("harvard.wav")

print(text)