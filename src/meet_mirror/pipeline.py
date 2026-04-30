from __future__ import annotations

import queue
import threading
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

from .asr import AsrWorker
from .audio_capture import AudioCapture
from .config import Config
from .persistence import AudioWriter, TextWriter
from .translator import TranslatorWorker
from .types import AudioChunk, EnSegment, ZhSegment


class Pipeline:
    AUDIO_Q_MAX = 200
    ASR_Q_MAX = 50
    SUBTITLE_Q_MAX = 50
    AUDIO_PERSIST_Q_MAX = 400
    SUBTITLE_PERSIST_Q_MAX = 100

    def __init__(self, config: Config) -> None:
        self.config = config

        # Live queues are persistent for the application lifetime so that the
        # SubtitleWindow holds a stable reference. We drain them on each
        # start() call to avoid replaying stale segments.
        self.audio_q: queue.Queue[AudioChunk] = queue.Queue(maxsize=self.AUDIO_Q_MAX)
        self.asr_q: queue.Queue[EnSegment] = queue.Queue(maxsize=self.ASR_Q_MAX)
        self.subtitle_q: queue.Queue[ZhSegment] = queue.Queue(
            maxsize=self.SUBTITLE_Q_MAX
        )

        # Persist queues are recreated per session.
        self.audio_q_persist: queue.Queue[AudioChunk] | None = None
        self.subtitle_q_persist: queue.Queue[ZhSegment] | None = None

        self.stop_event = threading.Event()
        self._workers: list[threading.Thread] = []
        self._running = False
        self._lock = threading.Lock()
        self.session_dir: Path | None = None
        # Updated by TranslatorWorker after each successful emit; consumed
        # by the status heartbeat. Plain int writes/reads are atomic in
        # CPython, no extra locking needed for observability.
        self.last_e2e_ms: int = 0
        self._heartbeat_thread: threading.Thread | None = None

    @staticmethod
    def _drain(q: queue.Queue) -> None:
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            return

    def _new_session_dir(self) -> Path:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        sessions_root = Path(self.config.session.dir)
        d = sessions_root / ts
        d.mkdir(parents=True, exist_ok=True)
        return d

    def start(self) -> None:
        with self._lock:
            self._start_locked()

    def _start_locked(self) -> None:
        if self._running:
            logger.warning("Pipeline already running")
            return
        self.stop_event.clear()
        self._drain(self.audio_q)
        self._drain(self.asr_q)
        self._drain(self.subtitle_q)

        self.session_dir = self._new_session_dir()
        session_start = time.time()
        logger.info(f"Session dir: {self.session_dir}")

        save_audio = self.config.session.save_audio
        save_text = self.config.session.save_transcript
        self.audio_q_persist = (
            queue.Queue(maxsize=self.AUDIO_PERSIST_Q_MAX) if save_audio else None
        )
        self.subtitle_q_persist = (
            queue.Queue(maxsize=self.SUBTITLE_PERSIST_Q_MAX) if save_text else None
        )

        # Worker startup order matters: ctranslate2 (Whisper) and
        # llama-cpp-python both want a CUDA context, and racing them
        # concurrently has been observed to deadlock CT2's loader. Spawn
        # ASR first, wait for Whisper to finish loading, *then* start the
        # rest. AudioCapture is also deferred so it doesn't burn capture
        # buffer while Whisper warms up.
        asr_ready = threading.Event()
        asr_worker = AsrWorker(
            in_q=self.audio_q,
            out_q=self.asr_q,
            stop_event=self.stop_event,
            sample_rate=self.config.audio.sample_rate,
            block_ms=self.config.audio.block_ms,
            asr_config=self.config.asr,
            ready_event=asr_ready,
        )
        asr_worker.start()
        if not asr_ready.wait(timeout=60.0):
            logger.warning(
                "ASR worker did not signal ready within 60s; "
                "continuing pipeline startup anyway"
            )
        else:
            logger.info("ASR worker ready; spawning translator + capture")

        translator_worker = TranslatorWorker(
            in_q=self.asr_q,
            out_q=self.subtitle_q,
            stop_event=self.stop_event,
            translator_config=self.config.translator,
            persist_q=self.subtitle_q_persist,
            pipeline_ref=self,
        )
        translator_worker.start()

        audio_worker = AudioCapture(
            out_q=self.audio_q,
            stop_event=self.stop_event,
            sample_rate=self.config.audio.sample_rate,
            block_ms=self.config.audio.block_ms,
            device=self.config.audio.capture_device,
            persist_q=self.audio_q_persist,
        )
        audio_worker.start()

        workers: list[threading.Thread] = [
            asr_worker,
            translator_worker,
            audio_worker,
        ]

        if self.audio_q_persist is not None:
            aw = AudioWriter(
                in_q=self.audio_q_persist,
                stop_event=self.stop_event,
                session_dir=self.session_dir,
                sample_rate=self.config.audio.sample_rate,
            )
            aw.start()
            workers.append(aw)
        if self.subtitle_q_persist is not None:
            tw = TextWriter(
                in_q=self.subtitle_q_persist,
                stop_event=self.stop_event,
                session_dir=self.session_dir,
                session_start=session_start,
            )
            tw.start()
            workers.append(tw)

        self._workers = workers
        self._running = True
        logger.info(f"Pipeline started with {len(self._workers)} workers")

        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="StatusHeartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        with self._lock:
            self._stop_locked(timeout)

    def _stop_locked(self, timeout: float = 5.0) -> None:
        if not self._running:
            return
        logger.info("Stopping pipeline...")
        self.stop_event.set()
        for w in self._workers:
            w.join(timeout=timeout)
            if w.is_alive():
                logger.warning(f"Worker {w.name} did not stop within {timeout}s")
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)
            self._heartbeat_thread = None
        self._workers = []
        self._running = False
        logger.info("Pipeline stopped")

    def _heartbeat_loop(self) -> None:
        """Emit a periodic status line so long-running sessions surface
        queue backpressure and end-to-end latency without the user
        having to grep DEBUG logs."""
        while not self.stop_event.is_set():
            if self.stop_event.wait(30.0):
                return
            try:
                logger.info(
                    f"status: audio_q={self.audio_q.qsize()} "
                    f"asr_q={self.asr_q.qsize()} "
                    f"sub_q={self.subtitle_q.qsize()} "
                    f"last_e2e_ms={self.last_e2e_ms}"
                )
            except Exception as e:
                logger.warning(f"Heartbeat read failed: {e}")

    @property
    def is_running(self) -> bool:
        return self._running
