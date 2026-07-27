# from faster_whisper import WhisperModel

# model_size = "medium.en"

# model = WhisperModel(
#     model_size,
#     device="cuda",
#     compute_type="int8"
# )

# segments, info = model.transcribe(
#     "hotwords.mp3",
#     beam_size=5
# )

# print("Detected language:", info.language)

# for segment in segments:
#     print(segment.text)



import os
import queue
import threading
import tkinter as tk
from datetime import datetime

import numpy as np
import soundcard as sc
from faster_whisper import WhisperModel


# ============================================================
# SETTINGS
# ============================================================

samplerate = 16000
chunk_duration = 1
channels = 1

frames_per_chunk = int(
    samplerate * chunk_duration
)


# ============================================================
# QUEUES / THREAD EVENTS
# ============================================================

audio_queue = queue.Queue()
text_queue = queue.Queue()

stop_event = threading.Event()
recording_done_event = threading.Event()


# ============================================================
# APPLICATION STATE
# ============================================================

listening = False

recording_thread = None
transcription_thread = None


# ============================================================
# TRANSCRIPT STORAGE
# ============================================================

# Contains ONLY actual Whisper transcription.
transcript_segments = []


# Tkinter Text positions corresponding to each transcript segment.
#
# Example:
# segment_positions[0] = ("4.0", "4.24")
segment_positions = []


# Everything before this index has already been copied.
copy_index = 0


# Beginning of current listening session.
session_start_index = 0


# ============================================================
# LAST / PREVIOUS COPIED TEXT
# ============================================================

# Stores the most recent block copied using "Copy New Text".
#
# The "Previous Copy" button will copy this exact text again.
last_copied_text = ""


# ============================================================
# WHISPER MODEL
# ============================================================

model = WhisperModel(
    "small.en",
    device="cuda",
    compute_type="int8_float16"
)


# ============================================================
# FIND JABRA SPEAKER
# ============================================================

def find_jabra_speaker():

    speakers = sc.all_speakers()

    print("\nAvailable speakers:")

    for i, speaker in enumerate(speakers):

        print(
            f"{i}: {speaker.name}"
        )

    for speaker in speakers:

        if "Jabra Engage 75" in speaker.name:

            print(
                f"\nSelected speaker: {speaker.name}"
            )

            return speaker

    raise RuntimeError(
        "Jabra Engage 75 speaker not found"
    )


# ============================================================
# AUDIO RECORDER
# ============================================================

def recorder():

    try:

        speaker = find_jabra_speaker()

        text_queue.put(
            (
                "system",
                f"Using speaker loopback: "
                f"{speaker.name}\n"
            )
        )


        with sc.get_microphone(
            id=str(speaker.name),
            include_loopback=True
        ).recorder(
            samplerate=samplerate,
            channels=channels
        ) as mic:


            text_queue.put(
                (
                    "system",
                    "Listening to Jabra "
                    "system audio...\n\n"
                )
            )


            while not stop_event.is_set():

                audio_data = mic.record(
                    numframes=frames_per_chunk
                )

                audio_queue.put(
                    audio_data.copy()
                )


    except Exception as e:

        text_queue.put(
            (
                "system",
                f"\nRecorder error: {e}\n"
            )
        )


    finally:

        # Tells transcription thread that recorder
        # has completely finished.
        recording_done_event.set()


# ============================================================
# WHISPER TRANSCRIPTION
# ============================================================

def transcriber():

    while True:

        # Stop only when:
        #
        # 1. recorder has finished
        # 2. audio queue is empty
        #
        if (
            recording_done_event.is_set()
            and audio_queue.empty()
        ):
            break


        try:

            audio_data = audio_queue.get(
                timeout=0.5
            )

        except queue.Empty:

            continue


        # Convert stereo/multichannel audio to mono.
        if audio_data.ndim > 1:

            audio_data = np.mean(
                audio_data,
                axis=1
            )


        audio_data = audio_data.astype(
            np.float32
        )


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

                    text_queue.put(
                        (
                            "transcript",
                            text
                        )
                    )


        except Exception as e:

            text_queue.put(
                (
                    "system",
                    f"\nTranscription error: "
                    f"{e}\n"
                )
            )


# ============================================================
# CLEAR AUDIO QUEUE
# ============================================================

def clear_audio_queue():

    while True:

        try:

            audio_queue.get_nowait()

        except queue.Empty:

            break


# ============================================================
# COPY BUTTON STATES
# ============================================================

def refresh_copy_buttons():

    # --------------------------------------------------------
    # COPY NEW TEXT
    # --------------------------------------------------------

    if copy_index < len(transcript_segments):

        copy_button.config(
            state=tk.NORMAL
        )

    else:

        copy_button.config(
            state=tk.DISABLED
        )


    # --------------------------------------------------------
    # PREVIOUS COPY
    # --------------------------------------------------------

    if last_copied_text:

        previous_copy_button.config(
            state=tk.NORMAL
        )

    else:

        previous_copy_button.config(
            state=tk.DISABLED
        )


# ============================================================
# PROCESS GUI TEXT QUEUE
# ============================================================

def process_text_queue():

    while True:

        try:

            message_type, text = (
                text_queue.get_nowait()
            )

        except queue.Empty:

            break


        # ----------------------------------------------------
        # ACTUAL TRANSCRIPTION
        # ----------------------------------------------------

        if message_type == "transcript":

            transcript_segments.append(
                text
            )


            # Tkinter position before inserting text.
            start_position = text_box.index(
                "end-1c"
            )


            text_box.insert(
                tk.END,
                text + "\n"
            )


            # Position after inserting transcript.
            end_position = text_box.index(
                "end-1c"
            )


            segment_positions.append(
                (
                    start_position,
                    end_position
                )
            )


            # New uncopied transcript exists.
            refresh_copy_buttons()


        # ----------------------------------------------------
        # SYSTEM MESSAGE
        # ----------------------------------------------------

        else:

            text_box.insert(
                tk.END,
                text
            )


        text_box.see(
            tk.END
        )


# ============================================================
# GUI UPDATE LOOP
# ============================================================

def update_text_box():

    process_text_queue()

    root.after(
        200,
        update_text_box
    )


# ============================================================
# START LISTENING
# ============================================================

def start_listening():

    global listening
    global recording_thread
    global transcription_thread
    global session_start_index
    global copy_index


    if listening:
        return


    process_text_queue()

    clear_audio_queue()


    stop_event.clear()

    recording_done_event.clear()

    listening = True


    # ========================================================
    # NEW SESSION
    # ========================================================

    session_start_index = len(
        transcript_segments
    )


    # Anything from before this session is considered
    # already handled for Copy New Text.
    copy_index = session_start_index


    # ========================================================
    # BUTTON STATES
    # ========================================================

    start_button.config(
        state=tk.DISABLED
    )

    save_button.config(
        state=tk.NORMAL
    )

    copy_button.config(
        state=tk.DISABLED
    )


    # Previous Copy remains available because it may contain
    # useful text copied immediately before starting
    # this new listening session.
    refresh_copy_buttons()


    status_label.config(
        text="Status: Listening..."
    )


    # ========================================================
    # SESSION HEADER
    # ========================================================

    current_time = datetime.now().strftime(
        "%H:%M:%S"
    )


    text_box.insert(
        tk.END,
        (
            f"\n--- New Listening Session "
            f"Started {current_time} ---\n"
        )
    )


    text_box.see(
        tk.END
    )


    # ========================================================
    # START RECORDING THREAD
    # ========================================================

    recording_thread = threading.Thread(
        target=recorder,
        daemon=True
    )


    # ========================================================
    # START TRANSCRIPTION THREAD
    # ========================================================

    transcription_thread = threading.Thread(
        target=transcriber,
        daemon=True
    )


    recording_thread.start()

    transcription_thread.start()


# ============================================================
# COPY ALL NEW / UNCOPIED TEXT
# ============================================================

def copy_new_text():

    global copy_index
    global last_copied_text


    # First process any Whisper results waiting in queue.
    process_text_queue()


    current_total = len(
        transcript_segments
    )


    # ========================================================
    # NO NEW TEXT
    # ========================================================

    if copy_index >= current_total:

        status_label.config(
            text=(
                "Status: No new transcription "
                "to copy"
            )
        )

        refresh_copy_buttons()

        return


    # ========================================================
    # DETERMINE NEW TEXT RANGE
    # ========================================================

    start_segment_index = copy_index

    end_segment_index = current_total - 1


    uncopied_segments = transcript_segments[
        start_segment_index:current_total
    ]


    # ========================================================
    # COMBINE NEW TEXT
    # ========================================================

    text_to_copy = " ".join(
        uncopied_segments
    ).strip()


    if not text_to_copy:

        status_label.config(
            text="Status: Nothing to copy"
        )

        return


    # ========================================================
    # COPY TO WINDOWS CLIPBOARD
    # ========================================================

    try:

        root.clipboard_clear()

        root.clipboard_append(
            text_to_copy
        )

        root.update_idletasks()


    except tk.TclError as e:

        status_label.config(
            text=f"Clipboard error: {e}"
        )

        return


    # ========================================================
    # STORE AS LAST / PREVIOUS COPY
    # ========================================================

    last_copied_text = text_to_copy


    # ========================================================
    # REMOVE PREVIOUS HIGHLIGHT
    # ========================================================

    # This is the important change.
    #
    # Previously all copied text remained highlighted.
    #
    # Now previous copied text returns to normal.
    #
    # Only the CURRENT newly copied block will be highlighted.
    text_box.tag_remove(
        "recent_copy",
        "1.0",
        tk.END
    )


    # ========================================================
    # HIGHLIGHT ONLY CURRENT COPY
    # ========================================================

    if (
        start_segment_index
        < len(segment_positions)
        and
        end_segment_index
        < len(segment_positions)
    ):


        start_position = (
            segment_positions[
                start_segment_index
            ][0]
        )


        end_position = (
            segment_positions[
                end_segment_index
            ][1]
        )


        text_box.tag_add(
            "recent_copy",
            start_position,
            end_position
        )


        text_box.see(
            end_position
        )


    # ========================================================
    # UPDATE COPY INDEX
    # ========================================================

    copied_count = (
        current_total - copy_index
    )


    copy_index = current_total


    status_label.config(
        text=(
            f"Status: Copied {copied_count} "
            f"new transcript segment(s)"
        )
    )


    refresh_copy_buttons()


# ============================================================
# COPY PREVIOUS / MOST RECENT COPIED BLOCK
# ============================================================

def copy_previous_text():

    # --------------------------------------------------------
    # NOTHING HAS BEEN COPIED YET
    # --------------------------------------------------------

    if not last_copied_text:

        status_label.config(
            text="Status: No previous copied text"
        )

        return


    # --------------------------------------------------------
    # COPY EXACT SAME BLOCK AGAIN
    # --------------------------------------------------------

    try:

        root.clipboard_clear()

        root.clipboard_append(
            last_copied_text
        )

        root.update_idletasks()


    except tk.TclError as e:

        status_label.config(
            text=f"Clipboard error: {e}"
        )

        return


    # IMPORTANT:
    #
    # Do NOT update copy_index.
    #
    # Do NOT change the highlight.
    #
    # The highlighted text remains the most recent block
    # created with "Copy New Text".
    status_label.config(
        text="Status: Previous copied text copied again"
    )


# ============================================================
# SAVE
# ============================================================

def save_text():

    global listening


    if not listening:
        return


    stop_event.set()

    listening = False


    # ========================================================
    # BUTTON STATES WHILE FINISHING WHISPER
    # ========================================================

    start_button.config(
        state=tk.DISABLED
    )

    save_button.config(
        state=tk.DISABLED
    )

    copy_button.config(
        state=tk.DISABLED
    )


    status_label.config(
        text=(
            "Status: Finishing final "
            "transcription..."
        )
    )


    threading.Thread(
        target=wait_for_threads_and_save,
        daemon=True
    ).start()


# ============================================================
# WAIT FOR RECORDER + WHISPER
# ============================================================

def wait_for_threads_and_save():

    # Wait for recorder to finish.
    if recording_thread is not None:

        recording_thread.join(
            timeout=5
        )


    # Wait for queued audio to finish transcription.
    if transcription_thread is not None:

        transcription_thread.join(
            timeout=15
        )


    try:

        root.after(
            0,
            finalize_save
        )

    except tk.TclError:

        pass


# ============================================================
# FINAL SAVE
# ============================================================

def finalize_save():

    # Process final Whisper output.
    process_text_queue()


    # ========================================================
    # CURRENT SESSION ONLY
    # ========================================================

    current_session = transcript_segments[
        session_start_index:
    ]


    # ========================================================
    # NO TRANSCRIPTION
    # ========================================================

    if not current_session:

        text_box.insert(
            tk.END,
            (
                "\n--- No transcription data "
                "to save ---\n"
            )
        )


        text_box.see(
            tk.END
        )


        start_button.config(
            state=tk.NORMAL
        )


        save_button.config(
            state=tk.DISABLED
        )


        refresh_copy_buttons()


        status_label.config(
            text="Status: No Data / Ready"
        )

        return


    # ========================================================
    # TRANSCRIPTS DIRECTORY
    # ========================================================

    script_directory = os.path.dirname(
        os.path.abspath(__file__)
    )


    transcript_directory = os.path.join(
        script_directory,
        "transcripts"
    )


    os.makedirs(
        transcript_directory,
        exist_ok=True
    )


    # ========================================================
    # DATE-BASED FILE
    # ========================================================

    now = datetime.now()


    date_string = now.strftime(
        "%Y-%m-%d"
    )


    file_name = (
        f"transcript_{date_string}.txt"
    )


    file_path = os.path.join(
        transcript_directory,
        file_name
    )


    # ========================================================
    # SESSION TIMESTAMP
    # ========================================================

    session_time = now.strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    # ========================================================
    # SAVE TRANSCRIPT
    # ========================================================

    try:

        transcript_text = " ".join(
            current_session
        ).strip()


        with open(
            file_path,
            "a",
            encoding="utf-8"
        ) as file:


            file.write(
                "\n\n"
            )


            file.write(
                "=" * 70
            )


            file.write(
                "\n"
            )


            file.write(
                f"Session: {session_time}\n"
            )


            file.write(
                "=" * 70
            )


            file.write(
                "\n\n"
            )


            file.write(
                transcript_text
            )


            file.write(
                "\n"
            )


    except Exception as e:

        text_box.insert(
            tk.END,
            f"\n--- Save Error: {e} ---\n"
        )


        text_box.see(
            tk.END
        )


        start_button.config(
            state=tk.NORMAL
        )


        refresh_copy_buttons()


        status_label.config(
            text="Status: Save Error"
        )

        return


    # ========================================================
    # SAVE CONFIRMATION
    # ========================================================

    text_box.insert(
        tk.END,
        (
            f"\n--- Transcript saved to "
            f"{file_name} ---\n"
        )
    )


    text_box.see(
        tk.END
    )


    start_button.config(
        state=tk.NORMAL
    )


    save_button.config(
        state=tk.DISABLED
    )


    # Final Whisper text may have arrived after the user
    # last clicked Copy New Text.
    refresh_copy_buttons()


    status_label.config(
        text=f"Status: Saved - {file_name}"
    )


# ============================================================
# CLOSE APPLICATION
# ============================================================

def on_close():

    stop_event.set()

    recording_done_event.set()

    root.destroy()


# ============================================================
# GUI
# ============================================================

root = tk.Tk()


root.title(
    "Jabra Real-Time Transcription App"
)


root.geometry(
    "1100x650"
)


# ============================================================
# TITLE
# ============================================================

title_label = tk.Label(
    root,
    text="Jabra Real-Time Speech Transcription",
    font=(
        "Arial",
        18,
        "bold"
    )
)


title_label.pack(
    pady=(15, 5)
)


# ============================================================
# STATUS
# ============================================================

status_label = tk.Label(
    root,
    text="Status: Ready",
    font=(
        "Arial",
        11
    )
)


status_label.pack(
    pady=(0, 5)
)


# ============================================================
# TEXT BOX FRAME
# ============================================================

text_frame = tk.Frame(
    root
)


text_frame.pack(
    padx=15,
    pady=10,
    fill=tk.BOTH,
    expand=True
)


# ============================================================
# SCROLLBAR
# ============================================================

scrollbar = tk.Scrollbar(
    text_frame
)


scrollbar.pack(
    side=tk.RIGHT,
    fill=tk.Y
)


# ============================================================
# TEXT BOX
# ============================================================

text_box = tk.Text(
    text_frame,
    wrap=tk.WORD,
    font=(
        "Arial",
        12
    ),
    yscrollcommand=scrollbar.set
)


text_box.pack(
    side=tk.LEFT,
    fill=tk.BOTH,
    expand=True
)


scrollbar.config(
    command=text_box.yview
)


# ============================================================
# RECENT COPY HIGHLIGHT
# ============================================================

# ONLY the most recent block copied through
# "Copy New Text" receives this highlight.
text_box.tag_configure(
    "recent_copy",
    background="#D9EAD3"
)


# ============================================================
# BUTTON FRAME
# ============================================================

button_frame = tk.Frame(
    root
)


button_frame.pack(
    pady=(5, 20)
)


# ============================================================
# START LISTENING
# ============================================================

start_button = tk.Button(
    button_frame,
    text="Start Listening",
    font=(
        "Arial",
        12
    ),
    width=18,
    height=2,
    command=start_listening
)


start_button.grid(
    row=0,
    column=0,
    padx=8
)


# ============================================================
# PREVIOUS COPY
# ============================================================

previous_copy_button = tk.Button(
    button_frame,
    text="Previous Copy",
    font=(
        "Arial",
        12
    ),
    width=18,
    height=2,
    command=copy_previous_text,
    state=tk.DISABLED
)


previous_copy_button.grid(
    row=0,
    column=1,
    padx=8
)


# ============================================================
# COPY NEW TEXT
# ============================================================

copy_button = tk.Button(
    button_frame,
    text="Copy New Text",
    font=(
        "Arial",
        12
    ),
    width=18,
    height=2,
    command=copy_new_text,
    state=tk.DISABLED
)


copy_button.grid(
    row=0,
    column=2,
    padx=8
)


# ============================================================
# SAVE
# ============================================================

save_button = tk.Button(
    button_frame,
    text="Save",
    font=(
        "Arial",
        12
    ),
    width=18,
    height=2,
    command=save_text,
    state=tk.DISABLED
)


save_button.grid(
    row=0,
    column=3,
    padx=8
)


# ============================================================
# CLOSE WINDOW
# ============================================================

root.protocol(
    "WM_DELETE_WINDOW",
    on_close
)


# ============================================================
# START GUI UPDATE
# ============================================================

update_text_box()


# ============================================================
# START APPLICATION
# ============================================================

root.mainloop()