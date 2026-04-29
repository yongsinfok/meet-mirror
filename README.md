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

> **Note:** This is the Slice 0 skeleton. It loads `config.yaml` and exits — there is no audio capture, ASR, or UI yet. Subsequent slices add those.

Requirements: Python 3.11+ (3.11 or 3.12 recommended). Later slices add CUDA-only dependencies.

```bash
git clone https://github.com/yongsinfok/meet-mirror.git
cd meet-mirror
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"
python main.py
# → Loaded config from config.yaml
# → Meet Mirror ready
```

To see the model artefacts that later slices will need:

```bash
python scripts/download_models.py
```

## Installation

_Full installation (with CUDA, Whisper, Qwen, PyQt) lands in Slice 2 onward. See the [Phase 1 plan](docs/superpowers/plans/2026-04-30-meet-mirror-phase1-plan.md)._

## License

TBD.
