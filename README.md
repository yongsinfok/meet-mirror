# Meet Mirror

A fully-local English → Chinese pipeline for Microsoft Teams meetings.

- **Phase 1 (`v0.1.0`)** — real-time floating-bar subtitle overlay during the meeting.
- **Phase 2 (`v0.2.1`)** — post-meeting Chinese Markdown summary with speaker labels.

100% on-device: no cloud APIs, no telemetry, no third-party services. Driven by Accenture client-data sensitivity — required, not preferred.

## What it does

### Phase 1 — live subtitles

During a Teams meeting:

1. Captures the system audio output (the *other* party's voice) via Windows WASAPI loopback.
2. Transcribes English speech in real time using `faster-whisper` (`large-v3-turbo`).
3. Translates each utterance into Chinese with a local `Qwen2.5-7B-Instruct` Q4_K_M GGUF on `llama-cpp-python`.
4. Renders the bilingual subtitle on a frameless, click-through, always-on-top Qt overlay.
5. Saves `audio.wav` + `transcript.txt` per session under `sessions/YYYY-MM-DD_HH-MM-SS/`.

End-to-end perceived latency: ~2.5 s for a 5-second utterance with natural pauses.

Tray icon and `Ctrl+Alt+T` global hotkey toggle the pipeline. Pipeline starts idle by default.

### Phase 2 — post-meeting notes

After the meeting, a one-shot CLI turns any saved session into a structured Chinese Markdown summary:

```bash
python -m meet_mirror_notes sessions/2026-04-30_13-20-51
```

Pipeline:

1. Re-transcribe `audio.wav` with `faster-whisper` (word-level timestamps).
2. Per-utterance voice embeddings via `Resemblyzer`, clustered with `sklearn` agglomerative clustering into N speakers (`--speakers`, default 2). Output: `transcript_with_speakers.json`.
3. Summarize via Qwen2.5-7B with the spec § 6.2 prompt structure (single-pass for ≤80-min sessions, automatic map-reduce beyond that). Output: `notes.md` with `概要 / 关键决策 / 行动项 / 未解决问题 / 时间线` sections.

## Hardware

| Item | Spec |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA, ≥ 16 GB VRAM, CUDA support |
| RAM | 64 GB recommended (Phase 1 alone runs in ≤16 GB system RAM) |
| Disk | ~10 GB free for model weights (`models/` + HF cache) |
| Python | 3.12 (3.13 / 3.14 lack ML wheels at time of writing) |

Tested on RTX PRO 4000 Blackwell Laptop GPU (sm_120, 16 GB). Steady-state numbers in [`BENCHMARKS.md`](BENCHMARKS.md).

## Architecture

```
                      Phase 1 (live)
Teams audio ──► audio_capture ──► asr ──► translator ──► subtitle_ui
   (WASAPI         │                                          │
    loopback)      ▼                                          ▼
              audio_writer                              text_writer
                   │                                          │
                   ▼                                          ▼
                                  sessions/<ts>/
                                  ├── audio.wav
                                  └── transcript.txt

                      Phase 2 (offline, on demand)
sessions/<ts>/audio.wav
   │
   ▼
asr_diarize.py ──► transcript_with_speakers.json
   │ (Whisper word timestamps + Resemblyzer + sklearn clustering)
   ▼
summarizer.py  ──► notes.md
   (Qwen2.5-7B Q4 single-pass or map-reduce)
```

Phase 1 is multi-threaded with bounded queues and drop-oldest backpressure; the Qt subtitle window lives on the main thread. Phase 2 is a single CLI invocation, models loaded sequentially on the GPU. Spec details in [`docs/superpowers/specs/2026-04-30-meet-mirror-design.md`](docs/superpowers/specs/2026-04-30-meet-mirror-design.md) and [`docs/superpowers/specs/2026-04-30-meet-mirror-phase2-notes-design.md`](docs/superpowers/specs/2026-04-30-meet-mirror-phase2-notes-design.md).

## Quick start

### 1. Create the venv

```bash
git clone https://github.com/yongsinfok/meet-mirror.git
cd meet-mirror
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

### 2. Install PyTorch with CUDA

PyTorch wheels live on its own index, not PyPI. Pick the wheel set that matches your GPU:

```bash
# RTX 50 / Blackwell (sm_120) — cu128 (torch 2.7+):
pip install torch --index-url https://download.pytorch.org/whl/cu128

# RTX 30 / 40 (sm_86 / sm_89) — cu124 (torch 2.4+) is sufficient:
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 3. Install llama-cpp-python with CUDA

PyPI ships only a CPU build. Install abetlen's prebuilt CUDA wheel (Python 3.12 / cu124 / Windows):

```bash
pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu124/llama_cpp_python-0.3.4-cp312-cp312-win_amd64.whl --force-reinstall --no-deps
```

To build from source instead (Visual Studio Build Tools + CUDA toolkit required):

```bash
set CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

(Substitute `CMAKE_CUDA_ARCHITECTURES=89` etc. for non-Blackwell.)

Verify CUDA is wired in:

```bash
python -c "from llama_cpp import llama_cpp; print(llama_cpp.llama_print_system_info().decode())"
# Should mention: ggml_cuda_init: found 1 CUDA devices
```

### 4. Install the project

```bash
pip install -e ".[dev]"
```

### 5. Download Qwen GGUF (~4.4 GB)

```bash
python scripts/download_models.py --qwen-only
# Pulls qwen2.5-7b-instruct-q4_k_m-{00001,00002}-of-00002.gguf into models/
```

Whisper `large-v3-turbo` (~1.5 GB) downloads to the HF cache automatically on first Phase 1 launch.

### 6. Run Phase 1

```bash
# Default — GUI mode (tray icon + floating subtitle bar):
python main.py
#   1. tray icon turns gray (idle) in the system tray
#   2. press Ctrl+Alt+T (or right-click tray → Start) to begin a session
#   3. ~10 s of model warm-up, then the bar shows EN + ZH on incoming Teams audio
#   4. Ctrl+Alt+T again to stop; sessions/<ts>/ written

# To reposition the subtitle bar once:
python main.py --unlock
#   - bar appears with a yellow border in drag mode
#   - drag to where you want it
#   - double-click to lock + persist position to config.yaml

# Console debug mode (terminal output, no GUI):
python main.py --mode console
```

Quit cleanly with the tray menu's **Quit**, `Ctrl+Q` while the bar is focused, or `Ctrl+C` in the terminal.

### 7. Run Phase 2 on a saved session

```bash
python -m meet_mirror_notes sessions/2026-04-30_13-20-51 --speakers 2
```

Produces `transcript_with_speakers.json` and `notes.md` next to the session's `audio.wav`. ~45 s wall time on a 3-min session; ~3–5 min for an hour-long meeting.

Useful flags:

- `--speakers N` — number of speakers (default 2)
- `--asr-only` / `--summary-only` — run just stage 1 or just stage 2
- `--force-asr` / `--force-summary` — re-run a stage even if its output exists
- `--n-ctx 32768` — bump the summarizer context for >2-hour meetings (default 16384, automatic map-reduce kicks in beyond ~80 min)

### 8. Verify the install

```bash
pytest -q
ruff check src/ tests/ scripts/
python scripts/capture_test.py                       # 10 s loopback + playback
python scripts/benchmark.py path/to/sample.wav       # Whisper latency / RTF
python scripts/benchmark.py --translator             # Qwen P50/P95 over 10 canned sentences
```

## Common gotchas

- **Translator silently dies during load.** Almost always a stuck `python.exe` from a previous session holding GPU memory. `tasklist | findstr python` then `taskkill /IM python.exe /F`. For verbose llama.cpp init output to triage when it doesn't go away, `set LLAMA_VERBOSE=1` before `python main.py`.
- **Whisper hangs at "Loading Whisper..." on a corp network.** `faster-whisper` does an HF HEAD/etag round-trip even with the cache populated. We auto-set `HF_HUB_OFFLINE=1` when the cache exists; if you want it explicit, `set HF_HUB_OFFLINE=1` before launch.
- **Loopback only catches the *other* party.** Your own microphone is excluded by design (the spec captures only the system audio output). For a sanity check that the audio path works, play a YouTube video — you should see EN lines within ~10 s.
- **Tray icon stays gray after toggle.** First fix: wait until you see `ASR worker ready` in the log (~10 s warm). If still gray, kill orphan python processes per the first bullet.

## Roadmap

| Version | Status | Scope |
|---|---|---|
| `v0.1.0` | shipped | Phase 1 — live EN→ZH overlay (S0–S6) |
| `v0.2.0` | shipped | Phase 2 alpha — single-pass summarizer |
| `v0.2.1` | current | Phase 2 — map-reduce + LLAMA_VERBOSE diagnostic |
| _next_ | TBD | follow-up work: e2e fixture wav, integration test, Phase 2 unit tests, prompt obedience tweaks |

The original Phase 1 spec mentioned `microsoft/VibeVoice-ASR-7B` for Phase 2 ASR + diarization. That repo doesn't exist publicly (Microsoft's VibeVoice is TTS-only), so Phase 2 pivoted to Whisper + Resemblyzer + sklearn clustering. See [Phase 2 spec § 6.1 + § 11](docs/superpowers/specs/2026-04-30-meet-mirror-phase2-notes-design.md) for details.

## License

TBD.
