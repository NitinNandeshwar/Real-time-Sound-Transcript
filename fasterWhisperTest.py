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

# Contains only actual Whisper transcription.
transcript_segments = []


# Stores Tkinter Text widget positions
# for every transcript segment.
#
# Used for highlighting Copy New Text.
segment_positions = []


# Everything before this index has already been copied
# using Copy New Text.
copy_index = 0


# Starting transcript index of the current listening session.
session_start_index = 0


# Stores the most recent block copied through Copy New Text.
#
# Previous Copy copies this same block again.
last_copied_text = ""


# ============================================================
# NORMAL COUNT-UP TIMER STATE
# ============================================================

# Tkinter after() callback ID.
#
# Used so an existing timer can be cancelled
# before restarting it.
timer_after_id = None


# Number of seconds elapsed since:
#
# Copy New Text
#
# OR
#
# Copy Full Transcript
#
# was last clicked.
timer_elapsed_seconds = 0


# ============================================================
# WHISPER MODEL
# ============================================================

model = WhisperModel(
    "small.en",    # "distil-small.en",
    device="cuda",
    compute_type="int8_float16"   # int8_float16
)


# ============================================================
# FORMAT NORMAL TIMER
# ============================================================

def format_timer(total_seconds):

    total_seconds = max(
        0,
        int(total_seconds)
    )


    hours, remainder = divmod(
        total_seconds,
        3600
    )


    minutes, seconds = divmod(
        remainder,
        60
    )


    # --------------------------------------------------------
    # LESS THAN ONE HOUR
    #
    # 00:00
    # 00:01
    # 00:59
    # 01:00
    # 59:59
    # --------------------------------------------------------

    if hours == 0:

        return (
            f"{minutes:02d}:"
            f"{seconds:02d}"
        )


    # --------------------------------------------------------
    # ONE HOUR OR MORE
    #
    # 01:00:00
    # 01:20:25
    # --------------------------------------------------------

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


# ============================================================
# UPDATE NORMAL TIMER
# ============================================================

def update_timer():

    global timer_after_id
    global timer_elapsed_seconds


    # --------------------------------------------------------
    # DISPLAY CURRENT ELAPSED TIME
    # --------------------------------------------------------

    timer_label.config(
        text=(
            "Copy Timer: "
            f"{format_timer(timer_elapsed_seconds)}"
        )
    )


    # --------------------------------------------------------
    # ADD ONE SECOND
    # --------------------------------------------------------

    timer_elapsed_seconds += 1


    # --------------------------------------------------------
    # UPDATE AGAIN AFTER ONE SECOND
    # --------------------------------------------------------

    timer_after_id = root.after(
        1000,
        update_timer
    )


# ============================================================
# RESET + START NORMAL TIMER
# ============================================================

def reset_copy_timer():

    global timer_after_id
    global timer_elapsed_seconds


    # --------------------------------------------------------
    # CANCEL EXISTING TIMER
    # --------------------------------------------------------

    if timer_after_id is not None:

        try:

            root.after_cancel(
                timer_after_id
            )

        except tk.TclError:

            pass


        timer_after_id = None


    # --------------------------------------------------------
    # RESET TO ZERO
    # --------------------------------------------------------

    timer_elapsed_seconds = 0


    # --------------------------------------------------------
    # START COUNTING
    # --------------------------------------------------------

    update_timer()


# ============================================================
# FIND JABRA SPEAKER
# ============================================================

def find_jabra_speaker():

    speakers = sc.all_speakers()


    print(
        "\nAvailable speakers:"
    )


    for i, speaker in enumerate(
        speakers
    ):

        print(
            f"{i}: {speaker.name}"
        )


    for speaker in speakers:

        if (
            "Jabra Engage 75"
            in speaker.name
        ):

            print(
                f"\nSelected speaker: "
                f"{speaker.name}"
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


        # ----------------------------------------------------
        # SHOW SELECTED SPEAKER
        # ----------------------------------------------------

        text_queue.put(
            (
                "system",
                (
                    "Using speaker loopback: "
                    f"{speaker.name}\n"
                )
            )
        )


        # ----------------------------------------------------
        # OPEN LOOPBACK RECORDING
        # ----------------------------------------------------

        with sc.get_microphone(
            id=str(
                speaker.name
            ),
            include_loopback=True
        ).recorder(
            samplerate=samplerate,
            channels=channels
        ) as mic:


            text_queue.put(
                (
                    "system",
                    (
                        "Listening to Jabra "
                        "system audio...\n\n"
                    )
                )
            )


            # ------------------------------------------------
            # RECORD AUDIO CONTINUOUSLY
            # ------------------------------------------------

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
                (
                    "\nRecorder error: "
                    f"{e}\n"
                )
            )
        )


    finally:

        # Recorder is completely finished.
        recording_done_event.set()


# ============================================================
# WHISPER TRANSCRIPTION
# ============================================================

def transcriber():

    while True:


        # ----------------------------------------------------
        # STOP ONLY WHEN:
        #
        # 1. Recorder has finished
        # 2. Audio queue is empty
        # ----------------------------------------------------

        if (
            recording_done_event.is_set()
            and
            audio_queue.empty()
        ):

            break


        # ----------------------------------------------------
        # GET AUDIO
        # ----------------------------------------------------

        try:

            audio_data = audio_queue.get(
                timeout=0.5
            )


        except queue.Empty:

            continue


        # ----------------------------------------------------
        # CONVERT TO MONO
        # ----------------------------------------------------

        if audio_data.ndim > 1:

            audio_data = np.mean(
                audio_data,
                axis=1
            )


        audio_data = audio_data.astype(
            np.float32
        )


        # ----------------------------------------------------
        # WHISPER
        # ----------------------------------------------------

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
                    (
                        "\nTranscription error: "
                        f"{e}\n"
                    )
                )
            )


# ============================================================
# CLEAR OLD AUDIO QUEUE
# ============================================================

def clear_audio_queue():

    while True:

        try:

            audio_queue.get_nowait()


        except queue.Empty:

            break


# ============================================================
# REFRESH COPY BUTTON STATES
# ============================================================

def refresh_copy_buttons():


    # --------------------------------------------------------
    # COPY NEW TEXT
    # --------------------------------------------------------

    if (
        copy_index
        <
        len(transcript_segments)
    ):

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


    # --------------------------------------------------------
    # COPY FULL TRANSCRIPT
    # --------------------------------------------------------

    if transcript_segments:

        full_copy_button.config(
            state=tk.NORMAL
        )


    else:

        full_copy_button.config(
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
        # WHISPER TRANSCRIPTION
        # ----------------------------------------------------

        if message_type == "transcript":


            # ------------------------------------------------
            # STORE TRANSCRIPT
            # ------------------------------------------------

            transcript_segments.append(
                text
            )


            # ------------------------------------------------
            # TEXT POSITION BEFORE INSERTING
            # ------------------------------------------------

            start_position = text_box.index(
                "end-1c"
            )


            # ------------------------------------------------
            # INSERT TRANSCRIPT
            # ------------------------------------------------

            text_box.insert(
                tk.END,
                text + "\n"
            )


            # ------------------------------------------------
            # TEXT POSITION AFTER INSERTING
            # ------------------------------------------------

            end_position = text_box.index(
                "end-1c"
            )


            # ------------------------------------------------
            # STORE POSITION
            # ------------------------------------------------

            segment_positions.append(
                (
                    start_position,
                    end_position
                )
            )


            refresh_copy_buttons()


        # ----------------------------------------------------
        # SYSTEM MESSAGE
        # ----------------------------------------------------

        else:

            text_box.insert(
                tk.END,
                text
            )


        # ----------------------------------------------------
        # SCROLL TO BOTTOM
        # ----------------------------------------------------

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


    # --------------------------------------------------------
    # ALREADY LISTENING
    # --------------------------------------------------------

    if listening:

        return


    # --------------------------------------------------------
    # PROCESS ANY WAITING TEXT
    # --------------------------------------------------------

    process_text_queue()


    # --------------------------------------------------------
    # CLEAR OLD AUDIO
    # --------------------------------------------------------

    clear_audio_queue()


    # --------------------------------------------------------
    # RESET EVENTS
    # --------------------------------------------------------

    stop_event.clear()

    recording_done_event.clear()


    listening = True


    # ========================================================
    # NEW SESSION
    # ========================================================

    session_start_index = len(
        transcript_segments
    )


    # Copy New Text starts from this session.
    #
    # Older transcripts still remain available through:
    #
    # Copy Full Transcript.

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


    refresh_copy_buttons()


    # ========================================================
    # SESSION HEADER
    # ========================================================

    current_time = datetime.now().strftime(
        "%H:%M:%S"
    )


    text_box.insert(
        tk.END,
        (
            "\n--- New Listening Session "
            f"Started {current_time} ---\n"
        )
    )


    text_box.see(
        tk.END
    )


    # ========================================================
    # RECORDING THREAD
    # ========================================================

    recording_thread = threading.Thread(
        target=recorder,
        daemon=True
    )


    # ========================================================
    # TRANSCRIPTION THREAD
    # ========================================================

    transcription_thread = threading.Thread(
        target=transcriber,
        daemon=True
    )


    # ========================================================
    # START THREADS
    # ========================================================

    recording_thread.start()

    transcription_thread.start()


# ============================================================
# COPY NEW TEXT
# ============================================================

def copy_new_text():

    global copy_index
    global last_copied_text


    # ========================================================
    # RESET NORMAL TIMER
    # ========================================================

    # Every time Copy New Text is clicked:
    #
    # 00:00
    # 00:01
    # 00:02
    # ...
    #
    # Timer restarts from zero.

    reset_copy_timer()


    # ========================================================
    # PROCESS LATEST WHISPER OUTPUT
    # ========================================================

    process_text_queue()


    current_total = len(
        transcript_segments
    )


    # ========================================================
    # NO NEW TEXT
    # ========================================================

    if copy_index >= current_total:

        refresh_copy_buttons()

        return


    # ========================================================
    # NEW TEXT RANGE
    # ========================================================

    start_segment_index = copy_index


    end_segment_index = (
        current_total - 1
    )


    uncopied_segments = (
        transcript_segments[
            start_segment_index:
            current_total
        ]
    )


    # ========================================================
    # COMBINE NEW TEXT
    # ========================================================

    text_to_copy = " ".join(
        uncopied_segments
    ).strip()


    if not text_to_copy:

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

        print(
            f"Clipboard error: {e}"
        )

        return


    # ========================================================
    # SAVE AS PREVIOUS COPY
    # ========================================================

    last_copied_text = text_to_copy


    # ========================================================
    # REMOVE OLD GREEN HIGHLIGHT
    # ========================================================

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
        <
        len(segment_positions)
        and
        end_segment_index
        <
        len(segment_positions)
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

    copy_index = current_total


    refresh_copy_buttons()


# ============================================================
# PREVIOUS COPY
# ============================================================

def copy_previous_text():


    # --------------------------------------------------------
    # NOTHING COPIED YET
    # --------------------------------------------------------

    if not last_copied_text:

        return


    # --------------------------------------------------------
    # COPY SAME BLOCK AGAIN
    # --------------------------------------------------------

    try:

        root.clipboard_clear()


        root.clipboard_append(
            last_copied_text
        )


        root.update_idletasks()


    except tk.TclError as e:

        print(
            f"Clipboard error: {e}"
        )


    # IMPORTANT:
    #
    # Previous Copy does NOT:
    #
    # - reset timer
    # - change copy_index
    # - change green highlight
    # - overwrite last_copied_text


# ============================================================
# COPY FULL TRANSCRIPT
# ============================================================

def copy_full_transcript():


    # ========================================================
    # RESET NORMAL TIMER
    # ========================================================

    # Every time Copy Full Transcript is clicked,
    # timer starts again from:
    #
    # 00:00

    reset_copy_timer()


    # ========================================================
    # PROCESS LATEST WHISPER OUTPUT
    # ========================================================

    process_text_queue()


    # ========================================================
    # NOTHING TRANSCRIBED
    # ========================================================

    if not transcript_segments:

        refresh_copy_buttons()

        return


    # ========================================================
    # CREATE FULL TRANSCRIPT
    # ========================================================

    # Everything from:
    #
    # First Whisper transcript
    #
    #             ↓
    #
    # Latest Whisper transcript

    full_transcript = " ".join(
        transcript_segments
    ).strip()


    if not full_transcript:

        return


    # ========================================================
    # COPY TO WINDOWS CLIPBOARD
    # ========================================================

    try:

        root.clipboard_clear()


        root.clipboard_append(
            full_transcript
        )


        root.update_idletasks()


    except tk.TclError as e:

        print(
            f"Clipboard error: {e}"
        )

        return


    # IMPORTANT:
    #
    # Copy Full Transcript does NOT:
    #
    # - change copy_index
    # - overwrite last_copied_text
    # - change Copy New Text green highlight


# ============================================================
# SAVE
# ============================================================

def save_text():

    global listening


    if not listening:

        return


    # --------------------------------------------------------
    # STOP RECORDING
    # --------------------------------------------------------

    stop_event.set()


    listening = False


    # ========================================================
    # BUTTON STATES
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


    # ========================================================
    # WAIT FOR FINAL TRANSCRIPTION
    # ========================================================

    threading.Thread(
        target=wait_for_threads_and_save,
        daemon=True
    ).start()


# ============================================================
# WAIT FOR RECORDER + WHISPER
# ============================================================

def wait_for_threads_and_save():


    # --------------------------------------------------------
    # WAIT FOR RECORDER
    # --------------------------------------------------------

    if recording_thread is not None:

        recording_thread.join(
            timeout=5
        )


    # --------------------------------------------------------
    # WAIT FOR WHISPER
    # --------------------------------------------------------

    if transcription_thread is not None:

        transcription_thread.join(
            timeout=15
        )


    # --------------------------------------------------------
    # RETURN TO TKINTER THREAD
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # PROCESS FINAL WHISPER OUTPUT
    # --------------------------------------------------------

    process_text_queue()


    # ========================================================
    # CURRENT SESSION
    # ========================================================

    current_session = transcript_segments[
        session_start_index:
    ]


    # ========================================================
    # NO DATA
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


        return


    # ========================================================
    # SCRIPT DIRECTORY
    # ========================================================

    script_directory = os.path.dirname(
        os.path.abspath(
            __file__
        )
    )


    # ========================================================
    # TRANSCRIPTS DIRECTORY
    # ========================================================

    transcript_directory = os.path.join(
        script_directory,
        "transcripts"
    )


    os.makedirs(
        transcript_directory,
        exist_ok=True
    )


    # ========================================================
    # DATE
    # ========================================================

    now = datetime.now()


    date_string = now.strftime(
        "%Y-%m-%d"
    )


    # ========================================================
    # FILE NAME
    # ========================================================

    file_name = (
        f"transcript_"
        f"{date_string}.txt"
    )


    file_path = os.path.join(
        transcript_directory,
        file_name
    )


    # ========================================================
    # SESSION TIME
    # ========================================================

    session_time = now.strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    # ========================================================
    # COMBINE CURRENT SESSION
    # ========================================================

    transcript_text = " ".join(
        current_session
    ).strip()


    # ========================================================
    # SAVE
    # ========================================================

    try:

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
                f"Session: "
                f"{session_time}\n"
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
            (
                "\n--- Save Error: "
                f"{e} ---\n"
            )
        )


        text_box.see(
            tk.END
        )


        start_button.config(
            state=tk.NORMAL
        )


        refresh_copy_buttons()


        return


    # ========================================================
    # SAVE CONFIRMATION
    # ========================================================

    text_box.insert(
        tk.END,
        (
            "\n--- Transcript saved to "
            f"{file_name} ---\n"
        )
    )


    text_box.see(
        tk.END
    )


    # ========================================================
    # BUTTON STATES
    # ========================================================

    start_button.config(
        state=tk.NORMAL
    )


    save_button.config(
        state=tk.DISABLED
    )


    refresh_copy_buttons()


# ============================================================
# CLOSE APPLICATION
# ============================================================

def on_close():

    global timer_after_id


    # --------------------------------------------------------
    # STOP AUDIO THREADS
    # --------------------------------------------------------

    stop_event.set()

    recording_done_event.set()


    # --------------------------------------------------------
    # CANCEL NORMAL TIMER
    # --------------------------------------------------------

    if timer_after_id is not None:

        try:

            root.after_cancel(
                timer_after_id
            )


        except tk.TclError:

            pass


        timer_after_id = None


    # --------------------------------------------------------
    # CLOSE GUI
    # --------------------------------------------------------

    root.destroy()


# ============================================================
# GUI
# ============================================================

root = tk.Tk()


root.title(
    "Jabra Real-Time Transcription App"
)


# Wider window because there are five buttons.

root.geometry(
    "1250x650"
)


# ============================================================
# TITLE
# ============================================================

title_label = tk.Label(
    root,
    text=(
        "Jabra Real-Time "
        "Speech Transcription"
    ),
    font=(
        "Arial",
        18,
        "bold"
    )
)


title_label.pack(
    pady=(
        15,
        5
    )
)


# ============================================================
# NORMAL COUNT-UP TIMER
# ============================================================

# Replaces the old Status text.
#
# Initial:
#
# Copy Timer: 00:00
#
# After Copy New Text / Copy Full Transcript:
#
# Copy Timer: 00:00
# Copy Timer: 00:01
# Copy Timer: 00:02
# Copy Timer: 00:03
# ...
# Copy Timer: 01:00
# Copy Timer: 01:01
#
# Clicking either copy button again resets it
# immediately to 00:00.

timer_label = tk.Label(
    root,
    text="Copy Timer: 00:00",
    font=(
        "Arial",
        14,
        # "bold"
    ),
    # fg="red"
)


timer_label.pack(
    pady=(
        0,
        5
    )
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

# Only the most recent Copy New Text block
# is highlighted in green.
#
# Previous Copy does not change it.
#
# Copy Full Transcript does not change it.

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
    pady=(
        5,
        20
    )
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
# COPY FULL TRANSCRIPT
# ============================================================

full_copy_button = tk.Button(
    button_frame,
    text="Copy Full Transcript",
    font=(
        "Arial",
        12
    ),
    width=18,
    height=2,
    command=copy_full_transcript,
    state=tk.DISABLED
)


full_copy_button.grid(
    row=0,
    column=3,
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
    column=4,
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