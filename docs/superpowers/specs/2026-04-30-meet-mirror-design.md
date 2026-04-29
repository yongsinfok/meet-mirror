# Meet Mirror — Design Spec

**Date:** 2026-04-30
**Status:** Draft, pending implementation plan
**Owner:** Joshua (yong-sin.fok@accenture.com)

---

## 1. Goals

Build a **fully-local** real-time English→Chinese translator for Microsoft Teams meetings, plus an offline post-meeting structured-notes pipeline.

**Primary use case (Phase 1 — real-time):**
- During a Teams meeting, capture the system audio (what the other party is saying in English).
- Display a Chinese translation on a floating subtitle bar at the bottom of the screen.
- Help the user understand colleagues / customers without needing perfect English listening comprehension.

**Secondary use case (Phase 2 — post-meeting):**
- After the meeting, take the saved recording and produce a structured Markdown summary using VibeVoice-ASR-7B (which provides speaker labels and timestamps), translated and organised by a local LLM.

**Hard constraints:**
- 100% local execution. No audio, transcript, or translation request may leave the machine. Cloud APIs (DeepL, OpenAI, Azure, Google) are explicitly forbidden because of Accenture client-data sensitivity.
- Hardware target: 16 GB VRAM, 64 GB RAM, Windows 11.

---

## 2. Non-Goals

- Voice cloning or voice changing (VibeVoice TTS is unrelated; we are not synthesising speech in Phase 1).
- Real-time TTS playback of the translation. The user reads subtitles, not listens.
- Translating the user's own speech (we capture system output only, not the microphone).
- Bilingual subtitles. Chinese only (per user preference).
- Cross-platform support. Windows 11 only.
- Translation of languages other than English source / Chinese target.
- VibeVoice-Realtime-0.5B usage. It is single-speaker preset-voice TTS and does not fit any goal here.

---

## 3. High-Level Architecture

```
                  ┌──────────────────────────────────────────────────────────┐
                  │                  Meet Mirror (Phase 1)                   │
                  │                                                          │
[Teams audio] ───►│ audio_capture ──[AudioChunk]──► asr ──[EnSegment]──┐    │
                  │     │                                              ▼    │
                  │     │                                         translator│
                  │     ▼                                              │    │
                  │  audio_q_persist                          [ZhSegment]   │
                  │     │                                              │    │
                  │     ▼                                              ▼    │
                  │  audio_writer                                subtitle_ui│──► [Floating bar]
                  │     │                                              │    │
                  │     ▼                                              ▼    │
                  │  audio.wav                                   text_writer│
                  │                                                    │    │
                  │                                                    ▼    │
                  │                                            transcript.txt
                  └──────────────────────────────────────────────────────────┘

                  ┌──────────────────────────────────────────────────────────┐
                  │             Post-Meeting Notes (Phase 2)                 │
                  │                                                          │
[audio.wav] ─────►│ vibevoice_asr_batch ─► transcript_with_speakers.json   │
                  │       │                          │                       │
                  │       │                          ▼                       │
                  │       │                 qwen_summarizer ─► notes.md      │
                  └──────────────────────────────────────────────────────────┘
```

**Communication:** Threading + `queue.Queue` (ML libraries are GIL-blocking C calls; asyncio buys nothing).

**4 worker classes (Phase 1):** `AudioCapture`, `AsrWorker`, `TranslatorWorker`, `SubtitleWindow` (Qt main thread). Two persistence side-workers: `AudioWriter`, `TextWriter`.

---

## 4. Project Layout

```
meet-mirror/
├── config.yaml
├── main.py
├── pyproject.toml
├── README.md
├── .gitignore
├── src/meet_mirror/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── audio_capture.py
│   ├── asr.py
│   ├── translator.py
│   ├── subtitle_ui.py
│   ├── persistence.py
│   ├── tray.py
│   └── types.py
├── src/meet_mirror_notes/                # Phase 2 (post-meeting), separate package
│   ├── __init__.py
│   ├── vibevoice_asr.py
│   └── summarizer.py
├── docs/superpowers/specs/               # Design specs (this file lives here)
├── sessions/                             # Auto-created session dirs (gitignored)
├── models/                               # GGUF + Whisper weights (gitignored)
├── logs/                                 # Loguru rotating logs (gitignored)
├── scripts/
│   ├── download_models.py
│   ├── benchmark.py
│   └── e2e_test.py
└── tests/
    ├── test_pipeline.py
    ├── test_translator.py
    ├── test_asr.py
    └── fixtures/                         # gitignored, user-recorded
```

---

## 5. Data Contracts (`src/meet_mirror/types.py`)

```python
from dataclasses import dataclass
from numpy import ndarray

@dataclass(frozen=True)
class AudioChunk:
    samples: ndarray         # float32 mono, 16 kHz
    ts_start: float          # monotonic seconds
    ts_end: float

@dataclass(frozen=True)
class EnSegment:
    text: str                # English transcription, stripped, single utterance
    audio_ts_start: float
    audio_ts_end: float

@dataclass(frozen=True)
class ZhSegment:
    en_text: str             # Original English (for transcript log)
    zh_text: str             # Chinese translation
    audio_ts_start: float
    audio_ts_end: float
    translation_latency_ms: int
```

---

## 6. Components

### 6.1 Audio Capture (`audio_capture.py`)

- Library: `soundcard` (Windows WASAPI loopback).
- Capture from `default_speaker()` with `include_loopback=True`.
- Samplerate 16 kHz, mono, 100 ms blocks (1600 samples).
- Each block emits an `AudioChunk` to two queues: `audio_q` (consumed by ASR) and `audio_q_persist` (consumed by `AudioWriter`).
- On stream error (device change, sleep/wake), reconnect via fresh `default_speaker()` lookup with exponential backoff (max 3 attempts before notifying user via tray toast).

### 6.2 ASR Worker (`asr.py`)

- Model: `faster-whisper` `large-v3-turbo`, FP16, CUDA.
- VAD: Silero VAD via faster-whisper's bundled implementation, threshold 0.5.
- Chunking strategy (state machine):
  - Accumulate voiced chunks into `voice_buffer`.
  - End-of-utterance: 500 ms continuous silence → flush.
  - Hard ceiling: 15 s of voice without silence → forced flush, continue accumulating.
  - Below 300 ms of voice → discard (likely noise).
- Transcription call:
  ```python
  segments, _ = model.transcribe(
      voice_buffer, language="en", vad_filter=False,
      beam_size=1, condition_on_previous_text=True,
  )
  ```
- Output: one `EnSegment` per flushed buffer to `asr_q`.

### 6.3 Translator Worker (`translator.py`)

- Model: Qwen2.5-7B-Instruct, GGUF Q4_K_M, via `llama-cpp-python`.
- Load: `n_gpu_layers=-1`, `n_ctx=8192`, `flash_attn=True`.
- System prompt:
  ```
  你是专业的实时翻译员,把英文商务会议对话翻译成中文。
  规则:
  - 只输出中文翻译,严禁输出原文、解释、引号、Markdown
  - 口语化,贴近商务会议场景
  - 专有名词、产品名、人名保留英文(如 Salesforce、AWS、John)
  - 不确定的词宁可保留英文,不要瞎猜
  ```
- Context: sliding window of last 5 (en, zh) pairs as few-shot examples (for style and pronoun consistency).
- Inference: `temperature=0.2`, `max_tokens=300`, `stop=["\n\n"]`. Non-streaming (whole-sentence translation arrives in <1 s).
- Output validation:
  - If input has fewer than 2 alphabetic chars → skip.
  - If output has zero CJK chars → retry once with `temperature=0.5`. Still none → emit empty `ZhSegment` and log warning.
- Output: `ZhSegment` to `subtitle_q` and `subtitle_q_persist`.

### 6.4 Subtitle UI (`subtitle_ui.py`)

- Toolkit: PyQt6, runs on the main thread.
- Window flags: `FramelessWindowHint | WindowStaysOnTopHint | Tool`, `WA_TranslucentBackground`.
- Style: rounded translucent black background, white text with 1 px black outline, default font "Microsoft YaHei" 24 pt (configurable).
- Position: bottom-centered by default, 80 px from screen bottom; draggable (position persisted to `config.yaml`).
- Display behaviour: 200 ms fade-in, hold 6 s after last update, fade-out.
- Mouse: transparent to clicks by default. Double-click subtitle → toggle "drag mode" (becomes opaque-bordered, can be repositioned), double-click again to lock.
- Polling: `QTimer` every 50 ms calls `subtitle_q.get_nowait()`.

### 6.5 Persistence (`persistence.py`)

- Session dir created when pipeline starts: `sessions/YYYY-MM-DD_HH-MM-SS/`.
- `AudioWriter` consumes `audio_q_persist`; writes int16 PCM into `audio.wav` via a single `soundfile.SoundFile(path, mode='w', samplerate=16000, channels=1, subtype='PCM_16')` opened at session start, calling `.write()` per chunk and `.close()` on session end (which finalises the WAV header).
- `TextWriter` consumes `subtitle_q_persist`; appends one line per segment:
  ```
  [00:01:23] EN: We need to align on the timeline before the demo. | ZH: 我们需要在 demo 之前对齐时间线。
  ```
- Both flush on session end.

### 6.6 Tray + Hotkey (`tray.py`)

- Tray icon via `pystray` (green = running, gray = idle).
- Right-click menu: `开始/停止`, `打开当前会话目录`, `重启`, `退出`.
- Global hotkey `Ctrl+Alt+T` via `keyboard` library — toggle pipeline start/stop.
  - Note: `keyboard` requires running as admin on Windows. If unavailable, fall back to `pynput` (no admin needed but slightly less reliable).

### 6.7 Pipeline Coordinator (`pipeline.py`)

- Owns all queues and worker instances.
- `start()` spawns worker threads; `stop()` signals shutdown via per-worker stop event, then joins (5 s timeout each).
- Subtitle UI is **not** in the worker pool — it lives on the Qt main thread because Qt requires it.

### 6.8 Main (`main.py`)

Startup sequence:
1. Load `config.yaml` (validated by Pydantic).
2. Verify model files in `models/`. If missing, print download instructions and exit.
3. Create `QApplication`.
4. Construct `Pipeline` (does not start yet).
5. Construct `SubtitleWindow` and bind to `subtitle_q`.
6. Start tray (background thread) and register hotkey.
7. Enter `app.exec()`.

---

## 7. Configuration (`config.yaml`)

```yaml
hotkey: "ctrl+alt+t"

audio:
  capture_device: default       # or device name string
  sample_rate: 16000
  block_ms: 100

asr:
  model: large-v3-turbo
  device: cuda
  compute_type: float16
  language: en
  vad_threshold: 0.5
  silence_ms_to_flush: 500
  max_utterance_s: 15

translator:
  model_path: models/qwen2.5-7b-instruct-q4_k_m.gguf
  n_gpu_layers: -1
  n_ctx: 8192
  temperature: 0.2
  max_tokens: 300
  history_size: 5

subtitle:
  font_family: "Microsoft YaHei"
  font_size: 24
  font_color: "#FFFFFF"
  background_color: "#000000"
  background_opacity: 0.7
  position: { x: null, y: null }   # null = auto bottom-center
  hold_seconds: 6

session:
  dir: sessions
  save_audio: true
  save_transcript: true

logging:
  level: INFO
  file: logs/meet-mirror.log
  rotation_mb: 10
```

---

## 8. Error Handling

| Failure | Detection | Response |
|---|---|---|
| Audio device disconnect / change | `soundcard` raises on next `record()` | Recreate stream from `default_speaker()`, exponential backoff up to 3 retries; tray toast on final failure |
| Whisper CUDA OOM | `RuntimeError` from `transcribe()` | One-time fall back to `medium` model with tray notification; pipeline continues |
| LLM inference >5 s | Timeout wrapper around `create_chat_completion` | Skip segment, log warning, do not block ASR queue |
| LLM returns no Chinese | CJK char count = 0 | Retry once at `temperature=0.5`; if still no Chinese, drop segment |
| Queue full (downstream too slow) | `Queue.put_nowait` raises `Full` | Drop oldest with `get_nowait`, log warning with queue name |
| Worker uncaught exception | Try/except at thread top level | Restart worker up to 3 times with 1 s back-off, then notify and run in degraded mode (other workers continue) |

**Logging:** Loguru, rotating 10 MB, level `INFO` by default (`DEBUG` available via config). Per-30-second status line: `asr_q=N sub_q=N last_e2e_ms=NNN`.

---

## 9. Performance Budget

| Stage | Expected | Budget |
|---|---|---|
| Audio capture (per 100 ms block) | <5 ms | 50 ms |
| VAD silence detection latency | 500 ms (config) | 500 ms |
| Whisper transcription (5 s utterance, large-v3-turbo) | ~700 ms | 1500 ms |
| Qwen2.5-7B translation (60-100 tokens) | ~600 ms | 1500 ms |
| UI render | <50 ms | 100 ms |
| **End-to-end perceived latency (5 s utterance)** | **~2 s** | **3.5 s** |

**VRAM:**
- Whisper large-v3-turbo FP16: ~1.6 GB
- Qwen2.5-7B Q4_K_M: ~5.0 GB
- KV cache + buffers: ~1.5 GB
- **Total: ~8.1 GB** out of 16 GB. Comfortable headroom for Phase 2 model swap.

---

## 10. Testing

| Layer | Scope | Tooling |
|---|---|---|
| Unit | `types.py`, sliding-window history, VAD state machine, CJK validator | pytest, pure logic, no GPU |
| Integration | ASR worker on fixture wav, Translator worker on fixture string | pytest, marked `@pytest.mark.gpu`, skipped in CI |
| End-to-end | 5-min English meeting wav → outputs in `sessions/test/` | `scripts/e2e_test.py`, manual quality check |
| Performance | Per-component latency on user's hardware | `scripts/benchmark.py`, output appended to `BENCHMARKS.md` |
| Real-world | Live Teams meetings | Dogfood, no automation |

**Fixtures:** `tests/fixtures/` is gitignored. The user records a 30 s English clip locally; never commit real meeting audio.

**No CI** in v1 — single-developer project, ML deps are heavy, gating value is low. Add later if collaborators join.

---

## 11. Phase 2 — Post-Meeting Notes (Brief)

After Phase 1 is shipped and stable, build `meet_mirror_notes` package:

1. `vibevoice_asr.py`: load `microsoft/VibeVoice-ASR-7B`, batch-transcribe a session's `audio.wav`, output structured JSON (speaker, ts, text). Estimated VRAM: ~14 GB Q4 — needs profiling on 16 GB. May require offloading some layers to CPU.
2. `summarizer.py`: feed structured transcript to Qwen2.5-7B with a "meeting notes" prompt (summary, action items, decisions, open questions, all in Chinese). Output `notes.md` next to the session's `audio.wav`.
3. CLI entry: `python -m meet_mirror_notes <session-dir>`.

Phase 2 has its own design doc (TBD) once Phase 1 is done.

---

## 12. Out of Scope / Explicitly Deferred

- LocalAgreement / true streaming Whisper (revisit if 2 s latency proves insufficient).
- Mobile / non-Windows platforms.
- Multi-language source (Spanish, Japanese, etc.).
- Speaker diarisation in real-time (only in Phase 2 batch).
- Translation memory / glossary persistence.
- Auto language detection.
- Web UI / remote viewing.

---

## 13. Open Questions

None at design time. All component choices, latency targets, and hardware budgets are settled.

---

## 14. Implementation Phasing (proposed for plan)

The plan should split Phase 1 into vertical slices that each end in a runnable system:

1. **Skeleton + audio capture + raw playback** — capture WASAPI loopback, play it back to confirm the audio path. No ML.
2. **+ ASR** — pipe captured audio through Whisper, print English to console.
3. **+ Translator** — add Qwen, print Chinese to console.
4. **+ Subtitle UI** — replace console with floating bar.
5. **+ Persistence + tray + hotkey** — productionise.
6. **+ Error handling + logging + benchmarks** — harden.

Each slice gets atomic commits and verification steps. Detailed plan to be produced by `superpowers:writing-plans`.
