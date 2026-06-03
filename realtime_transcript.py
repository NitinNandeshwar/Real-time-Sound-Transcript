import soundcard as sc
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel


# Settings
samplerate = 16000
chunk_duration = 1.5   # ✅ reduced from 2 seconds for faster response
channels = 1

frames_per_chunk = int(samplerate * chunk_duration)

audio_queue = queue.Queue()


model = WhisperModel(
    "small.en",
    device="cuda",
    compute_type="int8_float16"   # ✅ better for GPU than plain int8
)


def find_jabra_speaker():
    speakers = sc.all_speakers()

    print("Available speakers:")
    for i, speaker in enumerate(speakers):
        print(i, speaker.name)

    for speaker in speakers:
        if "Jabra Engage 75" in speaker.name:
            return speaker

    raise RuntimeError("Jabra Engage 75 speaker not found")


def recorder():
    speaker = find_jabra_speaker()

    print(f"🔊 Using speaker loopback: {speaker.name}")

    # ✅ loopback=True captures system sound playing through Jabra
    with sc.get_microphone(
        id=str(speaker.name),
        include_loopback=True
    ).recorder(samplerate=samplerate, channels=channels) as mic:

        print("🎧 Listening to Jabra system audio... Press Ctrl+C to stop.")

        while True:
            audio_data = mic.record(numframes=frames_per_chunk)
            audio_queue.put(audio_data.copy())


def transcriber():
    while True:
        audio_data = audio_queue.get()

        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        audio_data = audio_data.astype(np.float32)

        segments, _ = model.transcribe(
            audio_data,
            language="en",
            beam_size=1,
            vad_filter=True,              # ✅ skips silence
            condition_on_previous_text=False
        )

        for segment in segments:
            text = segment.text.strip()
            if text:
                print(text)


threading.Thread(target=recorder, daemon=True).start()
transcriber()