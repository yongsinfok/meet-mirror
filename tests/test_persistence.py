from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from meet_mirror.persistence import AudioWriter, TextWriter, _fmt_relative
from meet_mirror.types import AudioChunk, ZhSegment


def test_fmt_relative_pads_components():
    assert _fmt_relative(0.0, 0.0) == "00:00:00"
    assert _fmt_relative(3.7, 0.0) == "00:00:03"
    assert _fmt_relative(65.5, 0.0) == "00:01:05"
    assert _fmt_relative(3661.0, 0.0) == "01:01:01"
    assert _fmt_relative(1005.0, 1000.0) == "00:00:05"
    # negative deltas clamp to zero
    assert _fmt_relative(900.0, 1000.0) == "00:00:00"


def test_audio_writer_produces_valid_wav(tmp_path: Path):
    in_q: queue.Queue = queue.Queue()
    stop = threading.Event()
    writer = AudioWriter(in_q, stop, tmp_path, sample_rate=16000)
    writer.start()

    for i in range(10):
        samples = np.full(1600, 0.1, dtype=np.float32)
        in_q.put(
            AudioChunk(samples=samples, ts_start=i * 0.1, ts_end=(i + 1) * 0.1)
        )

    time.sleep(1.0)
    stop.set()
    writer.join(timeout=5)
    assert not writer.is_alive()

    wav_path = tmp_path / "audio.wav"
    assert wav_path.exists()
    data, sr = sf.read(str(wav_path))
    assert sr == 16000
    # 10 blocks * 1600 samples = 16000 samples = 1.0 s
    assert abs(len(data) / sr - 1.0) < 0.05
    # PCM_16 round-trip lands close to the input value (soundfile reads as
    # float in [-1, 1]); 0.1 -> int16 ~= 3277 -> back ~= 0.1
    assert 0.05 < float(np.mean(np.abs(data))) < 0.15


def test_text_writer_formats_relative_timestamps_and_bilingual_lines(
    tmp_path: Path,
):
    in_q: queue.Queue = queue.Queue()
    stop = threading.Event()
    writer = TextWriter(in_q, stop, tmp_path, session_start=1000.0)
    writer.start()

    in_q.put(
        ZhSegment(
            en_text="Hello world.",
            zh_text="你好，世界。",
            audio_ts_start=1003.0,
            audio_ts_end=1005.0,
            translation_latency_ms=500,
        )
    )
    in_q.put(
        ZhSegment(
            en_text="Second one.",
            zh_text="第二个。",
            audio_ts_start=1010.5,
            audio_ts_end=1012.0,
            translation_latency_ms=400,
        )
    )
    in_q.put(
        ZhSegment(
            en_text="Third.",
            zh_text="第三。",
            audio_ts_start=1085.0,
            audio_ts_end=1087.0,
            translation_latency_ms=300,
        )
    )

    time.sleep(1.0)
    stop.set()
    writer.join(timeout=5)
    assert not writer.is_alive()

    transcript = (tmp_path / "transcript.txt").read_text(encoding="utf-8")
    lines = transcript.strip().split("\n")
    assert lines == [
        "[00:00:03] EN: Hello world. | ZH: 你好，世界。",
        "[00:00:10] EN: Second one. | ZH: 第二个。",
        "[00:01:25] EN: Third. | ZH: 第三。",
    ]


def test_text_writer_drains_pending_items_when_stop_arrives(tmp_path: Path):
    """If stop_event fires while items are still queued, the post-loop
    drain pass must still flush them."""
    in_q: queue.Queue = queue.Queue()
    stop = threading.Event()
    writer = TextWriter(in_q, stop, tmp_path, session_start=0.0)

    # Pre-load the queue so items are guaranteed to be present when the
    # writer enters its drain phase.
    for i in range(3):
        in_q.put(
            ZhSegment(
                en_text=f"en{i}",
                zh_text=f"zh{i}",
                audio_ts_start=float(i),
                audio_ts_end=float(i) + 1.0,
                translation_latency_ms=0,
            )
        )
    stop.set()
    writer.start()
    writer.join(timeout=5)

    transcript = (tmp_path / "transcript.txt").read_text(encoding="utf-8")
    assert transcript.count("\n") == 3
