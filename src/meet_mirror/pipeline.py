from __future__ import annotations

import queue
import threading

from loguru import logger

from .asr import AsrWorker
from .audio_capture import AudioCapture
from .config import Config
from .translator import TranslatorWorker
from .types import AudioChunk, EnSegment, ZhSegment


class Pipeline:
    AUDIO_Q_MAX = 200
    ASR_Q_MAX = 50
    SUBTITLE_Q_MAX = 50

    def __init__(self, config: Config) -> None:
        self.config = config
        self.audio_q: queue.Queue[AudioChunk] = queue.Queue(maxsize=self.AUDIO_Q_MAX)
        self.asr_q: queue.Queue[EnSegment] = queue.Queue(maxsize=self.ASR_Q_MAX)
        self.subtitle_q: queue.Queue[ZhSegment] = queue.Queue(maxsize=self.SUBTITLE_Q_MAX)
        self.stop_event = threading.Event()
        self._workers: list[threading.Thread] = []
        self._running = False

    def start(self) -> None:
        if self._running:
            logger.warning("Pipeline already running")
            return
        self.stop_event.clear()
        self._workers = [
            AudioCapture(
                out_q=self.audio_q,
                stop_event=self.stop_event,
                sample_rate=self.config.audio.sample_rate,
                block_ms=self.config.audio.block_ms,
                device=self.config.audio.capture_device,
            ),
            AsrWorker(
                in_q=self.audio_q,
                out_q=self.asr_q,
                stop_event=self.stop_event,
                sample_rate=self.config.audio.sample_rate,
                block_ms=self.config.audio.block_ms,
                asr_config=self.config.asr,
            ),
            TranslatorWorker(
                in_q=self.asr_q,
                out_q=self.subtitle_q,
                stop_event=self.stop_event,
                translator_config=self.config.translator,
            ),
        ]
        for w in self._workers:
            w.start()
        self._running = True
        logger.info(f"Pipeline started with {len(self._workers)} workers")

    def stop(self, timeout: float = 5.0) -> None:
        if not self._running:
            return
        logger.info("Stopping pipeline...")
        self.stop_event.set()
        for w in self._workers:
            w.join(timeout=timeout)
            if w.is_alive():
                logger.warning(f"Worker {w.name} did not stop within {timeout}s")
        self._workers = []
        self._running = False
        logger.info("Pipeline stopped")
