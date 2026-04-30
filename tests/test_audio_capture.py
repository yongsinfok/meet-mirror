from __future__ import annotations

import queue
import threading
from unittest.mock import MagicMock, patch

import numpy as np

from meet_mirror.audio_capture import AudioCapture
from meet_mirror.types import AudioChunk


def test_audio_capture_emits_chunks_with_monotonic_timestamps():
    sample_rate = 16000
    block_ms = 100
    blocksize = sample_rate * block_ms // 1000  # 1600

    fake_blocks = [
        np.full((blocksize, 1), 0.1, dtype=np.float32),
        np.full((blocksize, 1), 0.2, dtype=np.float32),
        np.full((blocksize, 1), 0.3, dtype=np.float32),
    ]

    out_q: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    call_count = {"n": 0}

    def fake_record(numframes: int):
        idx = call_count["n"]
        call_count["n"] += 1
        if idx < len(fake_blocks):
            if idx == len(fake_blocks) - 1:
                stop_event.set()
            return fake_blocks[idx]
        return np.zeros((numframes, 1), dtype=np.float32)

    rec = MagicMock()
    rec.record.side_effect = fake_record

    cm = MagicMock()
    cm.__enter__.return_value = rec
    cm.__exit__.return_value = False

    mic = MagicMock()
    mic.recorder.return_value = cm
    mic.name = "fake-loopback"

    speaker = MagicMock()
    speaker.name = "fake-speaker"

    with patch("meet_mirror.audio_capture.sc") as sc_mock:
        sc_mock.default_speaker.return_value = speaker
        sc_mock.get_microphone.return_value = mic

        cap = AudioCapture(
            out_q=out_q,
            stop_event=stop_event,
            sample_rate=sample_rate,
            block_ms=block_ms,
        )
        cap.start()
        cap.join(timeout=2.0)

    assert not cap.is_alive(), "capture thread did not stop"

    chunks: list[AudioChunk] = []
    while not out_q.empty():
        chunks.append(out_q.get_nowait())

    assert len(chunks) == 3
    for c in chunks:
        assert isinstance(c, AudioChunk)
        assert c.samples.shape == (blocksize,)
        assert c.samples.dtype == np.float32

    # ts_start monotonic non-decreasing
    assert chunks[0].ts_start <= chunks[1].ts_start <= chunks[2].ts_start

    # block duration matches blocksize / sample_rate
    expected_dur = blocksize / sample_rate
    for c in chunks:
        assert abs((c.ts_end - c.ts_start) - expected_dur) < 1e-3


def test_audio_capture_uses_named_device():
    out_q: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    stop_event.set()  # exit immediately after recorder opens

    rec = MagicMock()
    rec.record.return_value = np.zeros((1600, 1), dtype=np.float32)

    cm = MagicMock()
    cm.__enter__.return_value = rec
    cm.__exit__.return_value = False

    mic = MagicMock()
    mic.recorder.return_value = cm
    mic.name = "named-loopback"

    speaker = MagicMock()
    speaker.name = "named-speaker"

    with patch("meet_mirror.audio_capture.sc") as sc_mock:
        sc_mock.get_speaker.return_value = speaker
        sc_mock.get_microphone.return_value = mic

        cap = AudioCapture(
            out_q=out_q,
            stop_event=stop_event,
            sample_rate=16000,
            block_ms=100,
            device="MyHeadphones",
        )
        cap.start()
        cap.join(timeout=2.0)

        sc_mock.get_speaker.assert_called_once_with("MyHeadphones")
        sc_mock.default_speaker.assert_not_called()
        sc_mock.get_microphone.assert_called_once_with(
            id=str(speaker.name), include_loopback=True
        )
