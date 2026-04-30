# Benchmarks

Hardware-specific latency and resource measurements. Append a new section per environment / slice.

---

## Slice 2 — Whisper ASR baseline

**Hardware:** NVIDIA RTX PRO 4000 Blackwell Laptop GPU (16 GB VRAM, sm_120), Windows 11.
**Stack:** Python 3.12.10, torch 2.11.0+cu128, faster-whisper 1.2.1, ctranslate2 4.7.1, silero-vad 6.2.1.
**Model:** `large-v3-turbo`, FP16, CUDA.

### Model load

| State | Time |
|---|---|
| Cold (first download + ct2 init) | 57.7 s |
| Warm (cache hit) | 8.3 s |

### Steady-state transcription

Input: 9.00 s wav (silent / music — produced one hallucinated token "Thank you").

| Metric | Value |
|---|---|
| Wall time | 254 ms |
| RTF | 0.028× |
| Tokens / sec | 19.6 (low; silent input) |

> Tokens/sec figure is not representative. Re-measure on a 30 s English speech wav once a real fixture is available.

### VRAM

Measured via `nvidia-smi` during live `--mode console` run:

| Process state | GPU memory (total) |
|---|---|
| Baseline (no Meet Mirror) | 1574 MiB |
| Meet Mirror running | 3840 MiB |
| **Delta (Meet Mirror)** | **~2266 MiB (~2.2 GB)** |

Spec §9 budget for Whisper FP16 alone is ~1.6 GB. Observed 2.2 GB includes ctranslate2 workspace, cuDNN handles, and silero-vad ONNX runtime. Acceptable given 16 GB total.

### End-to-end behaviour

- 1 m 45 s live Teams loopback → 8 EN segments printed.
- 15 s forced-flush ceiling triggers cleanly on continuous speech (no silence pauses in source clip).
- 500 ms silence flush untested in this run (no natural pauses); will validate during real Teams meeting dogfooding.
- Cold-start audio_q overflow drops ~37 s of audio while model loads (capture blocks on full queue). Slice 6 will switch to drop-oldest semantics.

---

## Slice 3 — TBD (Qwen translator)
## Slice 6 — TBD (full e2e + 60-min soak)
