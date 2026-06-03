from faster_whisper import WhisperModel

model_size = "medium.en"

model = WhisperModel(
    model_size,
    device="cuda",
    compute_type="int8"
)

segments, info = model.transcribe(
    "hotwords.mp3",
    beam_size=5
)

print("Detected language:", info.language)

for segment in segments:
    print(segment.text)