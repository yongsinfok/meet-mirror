from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundcard as sc
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meet_mirror.audio_capture import AudioCapture  # noqa: E402

SAMPLE_RATE = 16000
DURATION_S = 10
OUT_PATH = ROOT / "tests" / "fixtures" / "capture_test.wav"


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    out_q: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    cap = AudioCapture(out_q=out_q, stop_event=stop_event, sample_rate=SAMPLE_RATE, block_ms=100)
    cap.start()

    print(f"Capturing {DURATION_S}s of system loopback audio...")
    t0 = time.monotonic()
    chunks: list[np.ndarray] = []
    while time.monotonic() - t0 < DURATION_S:
        try:
            chunk = out_q.get(timeout=0.5)
        except queue.Empty:
            continue
        chunks.append(chunk.samples)

    stop_event.set()
    cap.join(timeout=2)

    while True:
        try:
            chunk = out_q.get_nowait()
        except queue.Empty:
            break
        chunks.append(chunk.samples)

    if not chunks:
        print("No audio captured. Is system audio playing?")
        return

    audio = np.concatenate(chunks).astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)
    sf.write(str(OUT_PATH), pcm16, SAMPLE_RATE, subtype="PCM_16")
    print(f"Wrote {OUT_PATH} ({len(audio) / SAMPLE_RATE:.2f}s)")

    print("Playing back...")
    data, sr = sf.read(str(OUT_PATH), dtype="float32")
    sc.default_speaker().play(data, samplerate=sr)
    print("Done.")


if __name__ == "__main__":
    main()
