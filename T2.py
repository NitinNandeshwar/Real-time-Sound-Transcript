import soundcard as sc
import numpy as np
import queue
import threading
import tkinter as tk
from tkinter import messagebox
from faster_whisper import WhisperModel


# ---------------- Settings ----------------
samplerate = 16000
chunk_duration = 1
channels = 1
frames_per_chunk = int(samplerate * chunk_duration)

audio_queue = queue.Queue()
text_queue = queue.Queue()

stop_event = threading.Event()
listening = False


# ---------------- Whisper Model ----------------
model = WhisperModel(
    "small.en",
    device="cuda",
    compute_type="int8_float16"
)


# ---------------- Audio Functions ----------------
def find_wh1000xm5_speaker():
    speakers = sc.all_speakers()

    print("Available speakers:")
    for i, speaker in enumerate(speakers):
        print(i, speaker.name)

    # Use device 0: Headphones (WH-1000XM5)
    speaker = speakers[0]

    if "WH-1000XM5" in speaker.name:
        return speaker

    raise RuntimeError("Device 0 is not WH-1000XM5. Found: " + speaker.name)


def recorder():
    try:
        speaker = find_wh1000xm5_speaker()
        text_queue.put(f"Using speaker loopback: {speaker.name}\n")

        with sc.get_microphone(
            id=str(speaker.name),
            include_loopback=True
        ).recorder(samplerate=samplerate, channels=channels) as mic:

            text_queue.put("Listening to WH-1000XM5 system audio...\n\n")

            while not stop_event.is_set():
                audio_data = mic.record(numframes=frames_per_chunk)
                audio_queue.put(audio_data.copy())

    except Exception as e:
        text_queue.put(f"\nRecorder error: {e}\n")


def transcriber():
    while not stop_event.is_set():
        try:
            audio_data = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        audio_data = audio_data.astype(np.float32)

        try:
            segments, _ = model.transcribe(
                audio_data,
                language="en",
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False
            )

            for segment in segments:
                text = segment.text.strip()
                if text:
                    text_queue.put(text + "\n")

        except Exception as e:
            text_queue.put(f"\nTranscription error: {e}\n")


# ---------------- GUI Functions ----------------
def start_listening():
    global listening

    if listening:
        return

    listening = True
    stop_event.clear()

    start_button.config(state=tk.DISABLED)
    save_button.config(state=tk.NORMAL)
    status_label.config(text="Status: Listening...")

    text_box.insert(tk.END, "\n--- New Listening Session Started ---\n")
    text_box.see(tk.END)

    threading.Thread(target=recorder, daemon=True).start()
    threading.Thread(target=transcriber, daemon=True).start()


def save_text():
    global listening

    stop_event.set()
    listening = False

    latest_text = text_box.get("1.0", tk.END).strip()

    if not latest_text:
        messagebox.showwarning("No Data", "No transcription data to save.")
        start_button.config(state=tk.NORMAL)
        save_button.config(state=tk.DISABLED)
        status_label.config(text="Status: Stopped")
        return

    file_path = "transcript1.txt"

    with open(file_path, "a", encoding="utf-8") as f:
        f.write("\n\n--- New Saved Session ---\n")
        f.write(latest_text)
        f.write("\n")

    messagebox.showinfo("Saved", "Transcript appended to transcript1.txt")

    text_box.insert(tk.END, "\n--- Latest text appended successfully ---\n")
    text_box.see(tk.END)

    start_button.config(state=tk.NORMAL)
    save_button.config(state=tk.DISABLED)
    status_label.config(text="Status: Saved / Ready")


def update_text_box():
    while not text_queue.empty():
        text = text_queue.get()
        text_box.insert(tk.END, text)
        text_box.see(tk.END)

    root.after(300, update_text_box)


def on_close():
    stop_event.set()
    root.destroy()


# ---------------- GUI Layout ----------------
root = tk.Tk()
root.title("WH-1000XM5 Real-Time Transcription App")
root.geometry("800x500")

title_label = tk.Label(
    root,
    text="WH-1000XM5 Real-Time Speech Transcription",
    font=("Arial", 16, "bold")
)
title_label.pack(pady=10)

status_label = tk.Label(
    root,
    text="Status: Ready",
    font=("Arial", 11)
)
status_label.pack()

text_box = tk.Text(
    root,
    wrap=tk.WORD,
    font=("Arial", 12),
    height=20
)
text_box.pack(padx=15, pady=10, fill=tk.BOTH, expand=True)

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

start_button = tk.Button(
    button_frame,
    text="Start Listening",
    font=("Arial", 12),
    width=18,
    command=start_listening
)
start_button.grid(row=0, column=0, padx=10)

save_button = tk.Button(
    button_frame,
    text="Save",
    font=("Arial", 12),
    width=18,
    command=save_text,
    state=tk.DISABLED
)
save_button.grid(row=0, column=1, padx=10)

root.protocol("WM_DELETE_WINDOW", on_close)

update_text_box()
root.mainloop()