from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioChunk:
    samples: np.ndarray
    ts_start: float
    ts_end: float


@dataclass(frozen=True)
class EnSegment:
    text: str
    audio_ts_start: float
    audio_ts_end: float


@dataclass(frozen=True)
class ZhSegment:
    en_text: str
    zh_text: str
    audio_ts_start: float
    audio_ts_end: float
    translation_latency_ms: int
