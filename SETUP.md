# Setup — fresh Windows 11 laptop, identical spec

This is a linear from-zero install for the **same hardware Joshua used**:

- Windows 11
- NVIDIA RTX PRO 4000 Blackwell Generation Laptop GPU (sm_120, 16 GB VRAM)
- 64 GB RAM
- ≥ 30 GB free disk

If your colleague has a *different* GPU (RTX 30/40, etc.), substitute the cu128 torch wheel with cu124 — see the README for the wheel selection table. Everything else applies as-is.

Estimated wall time: **~30–45 min** (most of that is downloads).

---

## 0. Things you need an account for

| Resource | Required? | Why |
|---|---|---|
| GitHub account | Yes | clone the repo |
| Hugging Face account | No | model downloads are public, no auth required |
| Admin rights on the laptop | Helpful | the `keyboard` package's global hotkey hook works without admin in most setups, but if `Ctrl+Alt+T` doesn't fire, you'll need to run as admin |

**Do not** create or use a Hugging Face token. Both models we use (`large-v3-turbo` for ASR, `Qwen2.5-7B-Instruct-GGUF` for translation) are public.

---

## 1. Install Python 3.12

The project requires **Python 3.12 specifically**. Do not use 3.13 or 3.14 — several ML wheels we need (`faster-whisper`'s `ctranslate2`, `llama-cpp-python`, `Resemblyzer`'s `librosa` chain) don't ship for them yet.

1. Go to https://www.python.org/downloads/release/python-31210/
2. Download **Windows installer (64-bit)**.
3. Run it. **Tick "Add python.exe to PATH"** before clicking Install Now.
4. Verify in a new `cmd.exe` window:

   ```cmd
   py -3.12 --version
   ```

   Should print `Python 3.12.10` (or similar 3.12.x).

If you already have 3.13/3.14 installed for other projects, that's fine — `py -3.12` selects the right one explicitly. Don't uninstall the others.

## 2. Install Git

If `git --version` doesn't already work in `cmd.exe`:

1. Download from https://git-scm.com/download/win
2. Run installer with default settings.

## 3. Install NVIDIA driver + verify GPU

You probably already have a working driver from when the laptop was provisioned. Verify:

```cmd
nvidia-smi
```

Expected output mentions `RTX PRO 4000 Blackwell` and `CUDA Version: 13.x` (or anything ≥ 12.4). If `nvidia-smi` is not found or fails, install the **NVIDIA Studio driver** for your GPU from https://www.nvidia.com/Download/index.aspx — choose your model and the latest "Studio" branch (not Game Ready). Reboot after install.

You **do not** need the standalone CUDA toolkit. Both `torch` (cu128 wheel) and `llama-cpp-python` (cu124 wheel) bundle their own CUDA runtime libraries.

## 4. Install Microsoft Visual C++ Redistributable

`faster-whisper` (via `ctranslate2`) and `llama-cpp-python` ship native DLLs that need the latest VC++ runtime. Most Windows 11 laptops already have it, but if anything fails with `ImportError: DLL load failed` later, install:

https://aka.ms/vs/17/release/vc_redist.x64.exe

(This is a 25 MB redistributable, not the full Visual Studio.)

## 5. Clone the repo and create the venv

Pick a location with ≥ 30 GB free space. Avoid OneDrive-synced folders — model weights inside the repo confuse OneDrive sync.

```cmd
cd C:\Project        :: or wherever you keep code
git clone https://github.com/yongsinfok/meet-mirror.git
cd meet-mirror
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

Every subsequent command in this guide assumes the venv is **activated** — you'll see `(.venv)` at the start of your prompt. If you open a new terminal later, re-run `.venv\Scripts\activate` first.

## 6. Install PyTorch with CUDA (Blackwell-compatible)

```cmd
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

This downloads ~2.5 GB. Verify CUDA is wired in:

```cmd
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `2.11.0+cu128 True NVIDIA RTX PRO 4000 Blackwell Generation Laptop GPU`

If `cuda.is_available()` prints `False`, your driver is too old or the install picked up the wrong wheel — re-check step 3.

## 7. Install llama-cpp-python with CUDA

PyPI's `llama-cpp-python` is CPU-only. We need abetlen's prebuilt CUDA wheel:

```cmd
pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu124/llama_cpp_python-0.3.4-cp312-cp312-win_amd64.whl --force-reinstall --no-deps
```

This downloads ~440 MB. Verify:

```cmd
python -c "from llama_cpp import llama_cpp; print(llama_cpp.llama_print_system_info().decode())"
```

Look for `ggml_cuda_init: found 1 CUDA devices` in the output. If you only see `CPU:` lines, the wrong wheel is installed — re-run the command above with `--force-reinstall`.

> The cu124 wheel ships kernels built for sm_500..sm_900. Blackwell sm_120 runs them via PTX JIT from sm_90 PTX. You'll see a small first-call JIT cost, then it runs at native speed.

## 8. Install the project

```cmd
pip install -e ".[dev]"
```

Pulls in the rest: `faster-whisper`, `silero-vad`, `PyQt6`, `keyboard`, `resemblyzer`, `scikit-learn`, `loguru`, etc. Total ~1 GB.

## 9. Download the Qwen translator (~4.4 GB)

```cmd
python scripts\download_models.py --qwen-only
```

Pulls two GGUF shards into `models\`:

```
qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf   (~4.0 GB)
qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf   (~0.7 GB)
```

The script logs the SHA256 of each file after download for identity verification.

> Whisper `large-v3-turbo` (~1.5 GB) auto-downloads on first Phase 1 run — you don't need to fetch it manually here.

## 10. Smoke-test the venv

```cmd
pytest -q
ruff check src/ tests/ scripts/
```

Expected:
- `21 passed`
- `All checks passed!`

If anything fails here, stop and report the error. Don't proceed.

## 11. First Phase 1 run (live subtitles)

```cmd
python main.py
```

What you'll see:

1. About 1 second after launch:
   ```
   Tray icon visible
   Global hotkey registered: ctrl+alt+t
   Meet Mirror ready (gui mode). ... Pipeline starts idle.
   ```
   A grey circle icon appears in your **system tray** (bottom-right of the taskbar — click the up-arrow `^` if it's hidden).

2. Press **`Ctrl+Alt+T`** (or right-click the tray icon → **Start**).
   First-ever launch downloads Whisper turbo (~1.5 GB). On subsequent launches the model loads in ~5–8 s.
   You'll see:
   ```
   Loading Whisper large-v3-turbo (cuda/float16)...
   ASR worker ready
   ASR worker ready; spawning translator + capture
   Loading Qwen translator: models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
   Translator ready
   Audio capture: loopback=Speakers ...
   ```
   Total cold start = ~15 s. Tray icon turns green.

3. Open any English audio in Teams, YouTube, or anywhere on your system. Within ~15 s a black floating bar appears with `EN: ...` (small grey text) above `ZH: ...` (large white text). The bar fades in, holds 6 seconds, fades out, ready for the next utterance.

4. Press **`Ctrl+Alt+T`** again to stop. The session writes `audio.wav` + `transcript.txt` into `sessions\YYYY-MM-DD_HH-MM-SS\`.

5. Quit the app via tray menu → **Quit**, or `Ctrl+C` in the terminal, or `Ctrl+Q` while focused on the bar.

### One-time bar repositioning

Default position is bottom-center. To move it:

```cmd
python main.py --unlock
```

The bar appears with a yellow border in **drag mode**. Drag it where you want it, then **double-click** the bar to lock it. Position is saved to `config.yaml`. Subsequent launches use that position.

## 12. First Phase 2 run (offline notes)

After you've done at least one Phase 1 session, run:

```cmd
python -m meet_mirror_notes sessions\2026-04-30_HH-MM-SS --speakers 2
```

Replace the path with the actual session directory name. The first run:

1. Re-transcribes `audio.wav` with Whisper (word-level timestamps). ~5 s per minute of audio.
2. Computes per-utterance voice embeddings with Resemblyzer + clusters into 2 speakers. ~3 s per minute.
3. Loads Qwen and summarizes via the spec § 6.2 prompt structure. ~30 s for short sessions, longer for hour-long meetings.

Outputs:

- `transcript_with_speakers.json` — speaker-labelled segments
- `notes.md` — Chinese summary with `概要 / 关键决策 / 行动项 / 未解决问题 / 时间线`

Total: about **45 s for a 3-min meeting**, **3–5 min for a 1-hour meeting**.

Re-running on the same session is cached — `transcript_with_speakers.json` and `notes.md` are skipped if they already exist. Use `--force-asr` and/or `--force-summary` to re-run a stage.

## 13. Common first-day errors

| Symptom | Cause | Fix |
|---|---|---|
| `python` not found | Path didn't pick up the install | Open a *new* terminal, or use `py -3.12` instead |
| `cuda.is_available() == False` | wrong torch wheel or stale driver | re-do step 6 (cu128 specifically), confirm `nvidia-smi` works |
| `ImportError: DLL load failed` on startup | missing VC++ runtime | install step 4 |
| Translator silently dies during load, no traceback | leftover `python.exe` from a previous crashed run holding GPU memory | `tasklist \| findstr python` then `taskkill /IM python.exe /F`, retry |
| Whisper hangs on `Loading Whisper...` for >1 min | corp network blocking HF metadata HEAD/etag call | `set HF_HUB_OFFLINE=1` then `python main.py`. The code auto-sets this when the cache exists, but explicit always wins |
| Tray icon stays grey after `Ctrl+Alt+T` | model load not finished yet | wait until log says `ASR worker ready` (~10 s warm). If still grey, kill orphan python processes |
| Bar never appears even though log shows `EN [...]: ...` and `ZH [...]: ...` | bar is invisible until the first segment arrives, but if it's positioned off-screen on a removed monitor you won't see it | run `python main.py --unlock` to reset to default bottom-center, drag, double-click to lock |
| `Ctrl+Alt+T` does nothing in any app | corporate security blocked the low-level keyboard hook | use the tray menu's **Start/Stop** instead, or run the terminal as admin |

## 14. What the colleague should know about loopback

Phase 1 captures **what your speakers are playing** (WASAPI loopback). It does **not** capture your microphone. So:

- The other party's voice → captured + translated ✓
- Your own voice into your mic → not captured (this is by design — the spec excludes the user's own speech because translating yourself is rarely useful)
- Teams group call where you're presenting and people are responding → all responses through your speakers are captured

If you join a meeting on a *headset* and the other party is on speaker on their end, their voice still routes through Teams → your headset → loopback. You do not need a separate "stereo mix" device.

For sanity check: open YouTube and play any English video. Within ~10–15 s of toggling on, you should see `EN [...]: ...` in the log. If not, your default audio output device may not be the one Teams uses — check Settings → System → Sound.

## 15. Updating

```cmd
cd C:\Project\meet-mirror
.venv\Scripts\activate
git pull
pip install -e ".[dev]"
```

Re-run `pip install` only when the changelog says new dependencies were added. Otherwise `git pull` is enough — the project is editable-installed.
