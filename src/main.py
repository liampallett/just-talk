import queue
import threading
import pyaudio
import numpy as np
import whisper

print("Loading Whisper...")
whisper_model = whisper.load_model("base")

CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

audio_queue = queue.Queue()

def _sanitise_text(s: str) -> list[str]:
    return [w.strip(".,!?").lower() for w in s.split() if w.strip(",.!?")]

def record_audio():
    """
    Threading: Continuously read mic data and push to queue for model to transcribe.
    :return: Nothing
    """
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_queue.put(data)

threading.Thread(target=record_audio, daemon=True).start()
audio_streaming_buffer = bytearray()

try:
    previous_text = ""
    while True:
        while not audio_queue.empty():
            audio_streaming_buffer.extend(audio_queue.get())

        if len(audio_streaming_buffer) >= 48000:
            raw_bytes = bytes(audio_streaming_buffer)
            audio_streaming_buffer = audio_streaming_buffer[24000:]

            audio_np = (
                np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
                / 32768.0
            )

            if np.sqrt(np.mean(audio_np**2)) > 0.01:
                result = whisper_model.transcribe(
                    audio_np,
                    fp16=False,
                    language="en",
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6
                )

                text = result.get("text", None).strip()
                if text:
                    curr_words = _sanitise_text(text)
                    prev_words = _sanitise_text(previous_text)

                    overlap_len = 0
                    for i in range(1, min(len(prev_words), len(curr_words)) + 1):
                        if prev_words[-i:] == curr_words[:i]:
                            overlap_len = i

                    if overlap_len > 0:
                        words = text.split()
                        text = " ".join(words[overlap_len:])

                    if text and text != previous_text:
                        print(f"{text}", end=" ", flush=True)
                        previous_text = result.get("text", "").strip()

except KeyboardInterrupt:
    print("\nStopping...")