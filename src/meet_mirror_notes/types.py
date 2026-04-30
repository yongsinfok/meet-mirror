from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptSegment:
    """One speaker-attributed utterance from the post-meeting ASR pass.

    Timestamps are session-relative seconds (start of audio.wav = 0.0).
    """

    speaker: str        # e.g. "Speaker 1", "Speaker 2"
    start_s: float
    end_s: float
    text: str
