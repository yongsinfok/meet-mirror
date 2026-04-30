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

## Slice 6 — error handling + observability

### Fault paths added

| Failure | Detection | Response |
|---|---|---|
| Audio device disconnect / change | exception from `recorder.record()` | `AudioCapture._capture_loop` exception → exponential backoff 1s→2s→4s, max 3 retries, then log error and exit worker (pipeline runs without input until tray toggle). |
| Whisper CUDA OOM | `RuntimeError` with `out of memory` substring from `model.transcribe()` | Drop the model, `torch.cuda.empty_cache()`, reload as `medium`. One-shot — never auto-upgrades back. |
| Whisper transcribe other failure | any other `RuntimeError` from `model.transcribe()` | Log with stack trace; skip current segment; loop continues. |
| Llama translation hang / >5s | `concurrent.futures` `TimeoutError` after 5 s on `create_chat_completion` | Skip segment; raise warning; ASR queue keeps draining without blocking. |
| Llama returns no CJK | post-decode CJK count = 0 | Retry once at `temperature=0.5`; if still empty, emit a `ZhSegment` with `zh_text=""` and warn. |
| Worker uncaught exception | top-level try/except wrapping `_run_inner()` | Log with stack; restart up to 3 times with 1 s → 4 s back-off; degraded mode after that. |
| Queue full (downstream slower than upstream) | `queue.Full` from `put_nowait` | `put_drop_oldest()` helper drops the oldest entry, logs the queue name, then re-enqueues the new item. Applied to `audio_q`, `audio_q_persist`, `asr_q`, `subtitle_q`, `subtitle_q_persist`. |

### Status heartbeat

Pipeline emits an INFO-level status line every 30 s with current queue depths and last observed end-to-end latency:

```
status: audio_q=N asr_q=N sub_q=N last_e2e_ms=NNN
```

`last_e2e_ms` is `time_emitted - audio_ts_end` for the most recently produced `ZhSegment`. With normal-pause English speech this should sit at 1500–3000 ms; persistent values >5000 ms indicate the LLM is keeping up with input but the perceived freshness has slipped (consider lowering `asr.max_utterance_s`).

### Deferred to a follow-up slice

- `scripts/e2e_test.py` against a checked-in 5-min English meeting wav fixture. Needs the user to record + commit the fixture (gitignored by default per `.gitignore` rule on `tests/fixtures/`), so blocked on a non-code step.
- `tests/test_pipeline.py` integration test driving fake AudioChunks through mocked ASR/Translator workers. Would replicate the existing dogfood path; lower ROI than the unit tests we already have for VadStateMachine, TranslatorWorker, and the persistence writers.
- Operator-visible UX for degraded mode (currently logged only). Will be added when a notification surface lands beyond the tray icon.

