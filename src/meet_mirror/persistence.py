from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger

from .types import AudioChunk, ZhSegment


def _fmt_relative(t_abs: float, session_start: float) -> str:
    delta = max(0.0, t_abs - session_start)
    h = int(delta // 3600)
    m = int((delta % 3600) // 60)
    s = int(delta % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class AudioWriter(threading.Thread):
    """Streams AudioChunks to <session_dir>/audio.wav as PCM_16 mono 16 kHz."""

    def __init__(
        self,
        in_q: queue.Queue[AudioChunk],
        stop_event: threading.Event,
        session_dir: Path,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__(name="AudioWriter", daemon=True)
        self.in_q = in_q
        self.stop_event = stop_event
        self.session_dir = Path(session_dir)
        self.sample_rate = sample_rate
        self.path = self.session_dir / "audio.wav"

    def _drain_one(self, sf_handle, chunk: AudioChunk) -> None:
        samples = np.asarray(chunk.samples, dtype=np.float32).reshape(-1)
        sf_handle.write(np.clip(samples, -1.0, 1.0))

    def run(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        with sf.SoundFile(
            str(self.path),
            mode="w",
            samplerate=self.sample_rate,
            channels=1,
            subtype="PCM_16",
        ) as f:
            while not self.stop_event.is_set():
                try:
                    chunk = self.in_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                self._drain_one(f, chunk)
            # Drain any remaining chunks before closing the WAV header.
            while True:
                try:
                    chunk = self.in_q.get_nowait()
                except queue.Empty:
                    break
                self._drain_one(f, chunk)
        logger.info(f"Wrote {self.path}")


class TextWriter(threading.Thread):
    """Appends one bilingual line per ZhSegment to <session_dir>/transcript.txt."""

    def __init__(
        self,
        in_q: queue.Queue[ZhSegment],
        stop_event: threading.Event,
        session_dir: Path,
        session_start: float,
    ) -> None:
        super().__init__(name="TextWriter", daemon=True)
        self.in_q = in_q
        self.stop_event = stop_event
        self.session_dir = Path(session_dir)
        self.session_start = session_start
        self.path = self.session_dir / "transcript.txt"

    def _format(self, seg: ZhSegment) -> str:
        ts = _fmt_relative(seg.audio_ts_start, self.session_start)
        return f"[{ts}] EN: {seg.en_text} | ZH: {seg.zh_text}\n"

    def run(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            while not self.stop_event.is_set():
                try:
                    seg = self.in_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                f.write(self._format(seg))
                f.flush()
            while True:
                try:
                    seg = self.in_q.get_nowait()
                except queue.Empty:
                    break
                f.write(self._format(seg))
        logger.info(f"Wrote {self.path}")
