from __future__ import annotations

import numpy as np

from meet_mirror.asr import VadStateMachine


def _block(val: float = 0.0, n: int = 1600) -> np.ndarray:
    return np.full(n, val, dtype=np.float32)


def test_silence_only_never_flushes():
    sm = VadStateMachine(block_ms=100, silence_ms_to_flush=500, max_utterance_s=15)
    for i in range(100):
        assert sm.feed(_block(), False, i * 0.1, (i + 1) * 0.1) is None
    assert not sm.in_utterance


def test_silence_flush_after_voice_emits_concatenated_audio():
    sm = VadStateMachine(
        block_ms=100, silence_ms_to_flush=500, max_utterance_s=15, min_voice_ms=300
    )
    # 5 voiced blocks
    for i in range(5):
        assert sm.feed(_block(0.5), True, i * 0.1, (i + 1) * 0.1) is None
    # 4 silent: no flush yet
    for i in range(5, 9):
        assert sm.feed(_block(), False, i * 0.1, (i + 1) * 0.1) is None
    # 5th silent (silence_run=5 == threshold) → flush
    out = sm.feed(_block(), False, 9 * 0.1, 10 * 0.1)
    assert out is not None
    # 5 voiced + 5 silent = 10 blocks, 1600 samples each
    assert out.audio.shape == (1600 * 10,)
    assert out.ts_start == 0.0
    assert abs(out.ts_end - 1.0) < 1e-9
    # state reset to IDLE
    assert not sm.in_utterance
    assert sm.feed(_block(), False, 1.0, 1.1) is None


def test_short_utterance_discarded():
    sm = VadStateMachine(
        block_ms=100, silence_ms_to_flush=500, max_utterance_s=15, min_voice_ms=300
    )
    # 2 voiced (200 ms < 300 ms min_voice)
    sm.feed(_block(0.5), True, 0.0, 0.1)
    sm.feed(_block(0.5), True, 0.1, 0.2)
    # 4 silent — no flush yet
    for i in range(2, 6):
        assert sm.feed(_block(), False, i * 0.1, (i + 1) * 0.1) is None
    # 5th silent triggers internal emit, but voice_blocks=2 < min_voice_blocks=3 → discard
    out = sm.feed(_block(), False, 0.6, 0.7)
    assert out is None
    # state reset; next voiced block starts new utterance
    assert sm.feed(_block(0.5), True, 0.7, 0.8) is None
    assert sm.in_utterance
    assert sm.voice_blocks == 1


def test_max_utterance_force_flush_at_15_seconds():
    sm = VadStateMachine(
        block_ms=100, silence_ms_to_flush=500, max_utterance_s=15, min_voice_ms=300
    )
    # 149 voiced — buffer not yet at max (150)
    for i in range(149):
        assert sm.feed(_block(0.5), True, i * 0.1, (i + 1) * 0.1) is None
    # 150th block → buffer length hits max_blocks → forced flush
    out = sm.feed(_block(0.5), True, 149 * 0.1, 150 * 0.1)
    assert out is not None
    assert out.audio.shape == (1600 * 150,)
    assert out.ts_start == 0.0
    assert abs(out.ts_end - 15.0) < 1e-6
    assert not sm.in_utterance


def test_voice_resumed_resets_silence_run():
    sm = VadStateMachine(
        block_ms=100, silence_ms_to_flush=500, max_utterance_s=15, min_voice_ms=300
    )
    for i in range(3):
        sm.feed(_block(0.5), True, i * 0.1, (i + 1) * 0.1)
    for i in range(3, 6):
        assert sm.feed(_block(), False, i * 0.1, (i + 1) * 0.1) is None
    assert sm.silence_run == 3
    # voice resumes — silence_run resets
    sm.feed(_block(0.5), True, 0.6, 0.7)
    assert sm.silence_run == 0
    assert sm.voice_blocks == 4
    assert sm.in_utterance


def test_invalid_block_ms_rejected():
    import pytest

    with pytest.raises(ValueError):
        VadStateMachine(block_ms=0)
