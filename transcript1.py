import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import tempfile
import time

SAMPLE_RATE = 16000
CHUNK_SECONDS = 5

model = WhisperModel("base", device="cpu", compute_type="int8")

def record_audio_chunk():
    audio = sd.rec(
        int(CHUNK_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    return audio

def transcribe(audio):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
        write(temp_file.name, SAMPLE_RATE, audio)
        segments, info = model.transcribe(temp_file.name)

        text = " ".join(segment.text for segment in segments)
        return text.strip()

while True:
    audio = record_audio_chunk()
    text = transcribe(audio)

    if text:
        print(text)

        with open("transcript.txt", "a", encoding="utf-8") as f:
            f.write(text + "\n")

    time.sleep(0.1)