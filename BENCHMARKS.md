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

## Slice 3 — Qwen translator

**Stack additions:** llama-cpp-python 0.3.4 (cu124 prebuilt wheel from abetlen GitHub release), Qwen2.5-7B-Instruct Q4_K_M GGUF (split: 3.99 GB + 0.69 GB).
**Llama config:** n_gpu_layers=-1 (all 29 layers on CUDA0), flash_attn=True, n_ctx=8192.

llama.cpp note: cu124 wheel ships kernels for sm_500..sm_900. Blackwell sm_120 runs via PTX JIT from sm_90 PTX — first call pays a JIT cost (~10 s), subsequent calls run at native speed.

### Model load (warm)

| Stage | Time |
|---|---|
| Qwen Q4_K_M load (Llama init + tensor offload) | 3.5 s |
| Whisper turbo (warm, after first session) | 7.8 s |
| Total pipeline cold start (Qwen + Whisper sequential) | ~16 s |

### Translator-only benchmark (10 canned business sentences, after warmup)

| Metric | Value |
|---|---|
| P50 | 426 ms |
| P95 | 821 ms |
| Mean | 424 ms |
| Min  | 266 ms |
| Max  | 821 ms |

Spec budget: P95 ≤ 1500 ms ✓.

### Live Teams loopback (3 min YouTube English clip about Multica)

| Metric | Value |
|---|---|
| Translator latency per ~60-char output | 1.8 – 2.7 s (mean ~2.2 s) |
| Perceived end-to-end (continuous speech, 15 s force-flush) | ~17 s after first word |
| Perceived end-to-end (estimated, natural pauses → 500 ms silence flush) | ~2.7 s |

The YouTube clip has near-zero pauses, so the 15 s force-flush dominates. A real meeting with sentence-level pauses should hit silence flush (500 ms) and stay within the 3.5 s spec budget.

### VRAM (live with both models loaded)

| State | GPU memory (total) |
|---|---|
| Baseline (Teams + Outlook + explorer) | 1574 MiB |
| Meet Mirror running (Whisper FP16 + Qwen Q4 + KV + buffers) | 8607 MiB |
| **Delta** | **~7033 MiB (~7 GB)** |

Spec §9 budget is ~8.1 GB target / 9 GB acceptance — measured 7 GB ✓.

### Translation quality observations

- Proper nouns preserved: Salesforce, AWS, VPS, Docker, Postgres, TypeScript, Next.js, API, Resend, OpenCode, Multica, .env.
- Style is conversational business as prompted; no quotes, no Markdown.
- "John" → "约翰" (Qwen translates common given names — acceptable per spec rule "宁可保留英文" applies to *uncertain* words, common names are not uncertain).
- Whisper mis-hearings propagate untouched (e.g. "Hetzner" → "Hesner", "Claude" → "Clawed").
- History window of 5 turns keeps "Multica" consistent across long technical narration.

---

## Slice 6 — TBD (full e2e + 60-min soak)
