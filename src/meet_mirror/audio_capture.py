from __future__ import annotations

import queue
import threading
import time

import numpy as np
import soundcard as sc
from loguru import logger

from .types import AudioChunk


class AudioCapture(threading.Thread):
    def __init__(
        self,
        out_q: queue.Queue[AudioChunk],
        stop_event: threading.Event,
        sample_rate: int = 16000,
        block_ms: int = 100,
        device: str = "default",
        persist_q: queue.Queue[AudioChunk] | None = None,
    ) -> None:
        super().__init__(name="AudioCapture", daemon=True)
        self.out_q = out_q
        self.stop_event = stop_event
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self.blocksize = int(sample_rate * block_ms / 1000)
        self.device = device
        self.persist_q = persist_q

    def _open_loopback(self):
        if self.device in (None, "", "default"):
            speaker = sc.default_speaker()
        else:
            speaker = sc.get_speaker(self.device)
        return sc.get_microphone(id=str(speaker.name), include_loopback=True)

    def run(self) -> None:
        mic = self._open_loopback()
        logger.info(
            f"Audio capture: loopback={mic.name} sr={self.sample_rate} block={self.block_ms}ms"
        )
        with mic.recorder(
            samplerate=self.sample_rate, channels=1, blocksize=self.blocksize
        ) as rec:
            while not self.stop_event.is_set():
                data = rec.record(numframes=self.blocksize)
                ts_end = time.time()
                ts_start = ts_end - self.blocksize / self.sample_rate
                samples = np.asarray(data, dtype=np.float32).reshape(-1)
                chunk = AudioChunk(
                    samples=samples, ts_start=ts_start, ts_end=ts_end
                )
                self.out_q.put(chunk)
                if self.persist_q is not None:
                    try:
                        self.persist_q.put_nowait(chunk)
                    except queue.Full:
                        # Persistence is best-effort; Slice 6 will tighten this.
                        pass
        logger.info("Audio capture stopped")
