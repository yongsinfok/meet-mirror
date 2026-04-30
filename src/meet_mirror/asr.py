from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
from loguru import logger

from .config import AsrConfig
from .types import AudioChunk, EnSegment


@dataclass
class _FlushReady:
    audio: np.ndarray
    ts_start: float
    ts_end: float


class VadStateMachine:
    """Pure-logic chunk-level VAD state machine.

    Per-block feed: voiced bool + samples. Emits a flush when:
      - silence_ms_to_flush of continuous silence after voice, or
      - buffer reaches max_utterance_s (forced flush).

    Drops utterances shorter than min_voice_ms (likely noise).
    """

    def __init__(
        self,
        block_ms: int = 100,
        silence_ms_to_flush: int = 500,
        max_utterance_s: int = 15,
        min_voice_ms: int = 300,
    ) -> None:
        if block_ms <= 0:
            raise ValueError("block_ms must be positive")
        self.block_ms = block_ms
        self.silence_blocks_to_flush = max(1, silence_ms_to_flush // block_ms)
        self.max_blocks = max(1, (max_utterance_s * 1000) // block_ms)
        self.min_voice_blocks = max(1, min_voice_ms // block_ms)
        self._reset()

    def _reset(self) -> None:
        self.in_utterance = False
        self.buffer: list[np.ndarray] = []
        self.voice_blocks = 0
        self.silence_run = 0
        self.first_ts = 0.0
        self.last_ts = 0.0

    def feed(
        self,
        samples: np.ndarray,
        voiced: bool,
        ts_start: float,
        ts_end: float,
    ) -> _FlushReady | None:
        if not self.in_utterance:
            if not voiced:
                return None
            self.in_utterance = True
            self.buffer = [samples]
            self.voice_blocks = 1
            self.silence_run = 0
            self.first_ts = ts_start
            self.last_ts = ts_end
            return None

        self.buffer.append(samples)
        self.last_ts = ts_end
        if voiced:
            self.voice_blocks += 1
            self.silence_run = 0
        else:
            self.silence_run += 1

        if len(self.buffer) >= self.max_blocks:
            return self._emit()

        if self.silence_run >= self.silence_blocks_to_flush:
            return self._emit()

        return None

    def _emit(self) -> _FlushReady | None:
        if self.voice_blocks < self.min_voice_blocks:
            self._reset()
            return None
        audio = np.concatenate(self.buffer).astype(np.float32)
        out = _FlushReady(audio=audio, ts_start=self.first_ts, ts_end=self.last_ts)
        self._reset()
        return out


class SileroVad:
    """Per-block voiced detector using Silero VAD ONNX model.

    Accepts arbitrary block lengths; slices into 512-sample windows
    (~32 ms at 16 kHz), takes max probability, thresholds.
    """

    WINDOW = 512
    SR = 16000

    def __init__(self, threshold: float = 0.5) -> None:
        import torch  # noqa: F401  (silero ONNX still uses torch tensors)
        from silero_vad import load_silero_vad

        self._torch = __import__("torch")
        self.model = load_silero_vad(onnx=True)
        self.threshold = threshold

    def is_voiced(self, samples: np.ndarray) -> bool:
        if samples.size < self.WINDOW:
            return False
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        max_prob = 0.0
        for i in range(0, samples.size - self.WINDOW + 1, self.WINDOW):
            window = samples[i : i + self.WINDOW]
            tensor = self._torch.from_numpy(window).float()
            prob = float(self.model(tensor, self.SR).item())
            if prob > max_prob:
                max_prob = prob
        return max_prob >= self.threshold


class AsrWorker(threading.Thread):
    def __init__(
        self,
        in_q: queue.Queue[AudioChunk],
        out_q: queue.Queue[EnSegment],
        stop_event: threading.Event,
        sample_rate: int,
        block_ms: int,
        asr_config: AsrConfig,
    ) -> None:
        super().__init__(name="AsrWorker", daemon=True)
        self.in_q = in_q
        self.out_q = out_q
        self.stop_event = stop_event
        self.sample_rate = sample_rate
        self.asr_config = asr_config
        self.vad_state = VadStateMachine(
            block_ms=block_ms,
            silence_ms_to_flush=asr_config.silence_ms_to_flush,
            max_utterance_s=asr_config.max_utterance_s,
            min_voice_ms=300,
        )

    def run(self) -> None:
        from faster_whisper import WhisperModel

        logger.info(
            f"Loading Whisper {self.asr_config.model} "
            f"({self.asr_config.device}/{self.asr_config.compute_type})..."
        )
        whisper = WhisperModel(
            self.asr_config.model,
            device=self.asr_config.device,
            compute_type=self.asr_config.compute_type,
        )
        vad = SileroVad(threshold=self.asr_config.vad_threshold)
        logger.info("ASR worker ready")

        while not self.stop_event.is_set():
            try:
                chunk = self.in_q.get(timeout=0.5)
            except queue.Empty:
                continue

            voiced = vad.is_voiced(chunk.samples)
            flush = self.vad_state.feed(
                chunk.samples, voiced, chunk.ts_start, chunk.ts_end
            )
            if flush is None:
                continue

            t0 = time.perf_counter()
            segments, _ = whisper.transcribe(
                flush.audio,
                language=self.asr_config.language,
                vad_filter=False,
                beam_size=1,
                condition_on_previous_text=True,
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            elapsed_ms = int((time.perf_counter() - t0) * 1000)

            if not text:
                logger.debug(
                    f"Empty transcription "
                    f"(elapsed={elapsed_ms}ms dur={flush.ts_end - flush.ts_start:.1f}s)"
                )
                continue

            seg = EnSegment(
                text=text,
                audio_ts_start=flush.ts_start,
                audio_ts_end=flush.ts_end,
            )
            logger.debug(
                f"EN segment: {text!r} "
                f"(transcribe={elapsed_ms}ms dur={flush.ts_end - flush.ts_start:.1f}s)"
            )
            self.out_q.put(seg)

        logger.info("ASR worker stopped")
