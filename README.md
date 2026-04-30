# Meet Mirror

A fully-local real-time English-to-Chinese subtitle overlay for Microsoft Teams meetings.

## Status

**Early development.** Design phase complete — see [`docs/superpowers/specs/2026-04-30-meet-mirror-design.md`](docs/superpowers/specs/2026-04-30-meet-mirror-design.md). Implementation has not started yet.

## What it does

During a Teams meeting, Meet Mirror:

1. Captures the system audio output (what the other party says) via Windows WASAPI loopback.
2. Transcribes English speech in real time using `faster-whisper` (large-v3-turbo).
3. Translates each utterance into Chinese with a local `Qwen2.5-7B-Instruct` model running on `llama.cpp`.
4. Displays the Chinese translation as a floating subtitle bar at the bottom of the screen.
5. Saves the meeting recording (`audio.wav`) and bilingual transcript (`transcript.txt`) to a session folder so you can produce structured notes afterwards (Phase 2).

End-to-end perceived latency is around 2 seconds for a 5-second utterance.

## Why fully-local

This project runs **entirely on your machine**. No audio, transcripts, or translation requests ever leave the host. There are no cloud API calls (no DeepL, no OpenAI, no Azure, no Google), no telemetry, no third-party services.

This is a hard requirement, not a preference, because the meetings being translated may contain client-confidential material.

## Hardware requirements

- Windows 11
- NVIDIA GPU with 16 GB VRAM and CUDA support
- 64 GB RAM
- ~10 GB free disk for model weights

## Architecture overview

```
Teams audio
   │ (WASAPI loopback)
   ▼
audio_capture ──► asr (faster-whisper) ──► translator (Qwen2.5-7B) ──► subtitle_ui (PyQt6)
                                                                            │
                                                                            ▼
                                                                       sessions/<ts>/
                                                                       ├── audio.wav
                                                                       └── transcript.txt
```

Four worker threads communicate through queues; the subtitle UI runs on the Qt main thread. Full architecture, data contracts, error handling, and performance budget are in the design spec linked above.

## Roadmap

- **Phase 1** (current): real-time EN→ZH subtitle overlay.
- **Phase 2** (planned): offline post-meeting structured notes using `VibeVoice-ASR-7B` with speaker labels and timestamps, summarised by Qwen2.5 into Chinese Markdown.

## Quick start

> **Status:** Slice 2 in progress. WASAPI loopback capture + Whisper ASR work in `--mode console`. Translator (Slice 3) and subtitle UI (Slice 4) not yet implemented.

Requirements: Python 3.12 (3.13/3.14 lack ML wheels at the time of writing), NVIDIA GPU + driver supporting CUDA 12.8, ~3 GB free disk for the Whisper model cache.

### 1. Create the venv

```bash
git clone https://github.com/yongsinfok/meet-mirror.git
cd meet-mirror
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

### 2. Install PyTorch with CUDA support

PyTorch wheels are not on PyPI; install them from the official PyTorch index **before** the editable install. Use the wheel set that matches your GPU:

```bash
# RTX 50 / Blackwell (sm_120) — needs cu128 wheels (torch 2.7+):
pip install torch --index-url https://download.pytorch.org/whl/cu128

# RTX 30 / 40 (sm_86 / sm_89) — cu124 wheels (torch 2.4+) are sufficient:
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Verify CUDA works:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 3. Install the project

```bash
pip install -e ".[dev]"
```

### 4. Run

```bash
# Slice 2 — console transcription (English only):
python main.py --mode console
#   Plays Teams audio with English speech →
#   [00:00:03] EN: We need to align on the timeline before the demo.

# Slice 0/1 — config check (no pipeline):
python main.py
```

The first run downloads the Whisper `large-v3-turbo` model (~1.5 GB) into the Hugging Face cache.

### 5. Verify

```bash
pytest -q
ruff check src/ tests/ scripts/
python scripts/capture_test.py            # 10 s loopback record + playback
python scripts/benchmark.py path/to/sample.wav  # Whisper latency / RTF
```

## Installation notes

_Translator (Qwen2.5-7B GGUF + llama-cpp-python) lands in Slice 3. Subtitle UI (PyQt6) in Slice 4. Tray + hotkey in Slice 5. See the [Phase 1 plan](docs/superpowers/plans/2026-04-30-meet-mirror-phase1-plan.md)._

## License

TBD.
