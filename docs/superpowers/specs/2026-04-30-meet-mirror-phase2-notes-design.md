# Meet Mirror — Phase 2 Notes Pipeline — Design Spec

**Date:** 2026-04-30
**Status:** Draft, depends on Phase 1 v0.1.0 stable
**Owner:** Joshua (yong-sin.fok@accenture.com)
**Related:** `docs/superpowers/specs/2026-04-30-meet-mirror-design.md` §11 (brief outline)

---

## 1. Goals

Take a session captured by Phase 1 (`sessions/<ts>/audio.wav` + `transcript.txt`) and produce a structured Chinese-language Markdown summary of the meeting, fully offline, with speaker labels.

**Primary deliverables (per session):**
- `notes.md` — human-readable Chinese meeting notes (summary, decisions, action items, open questions, full timeline).
- `transcript_with_speakers.json` — structured intermediate (speaker, start_s, end_s, text) for downstream tooling and re-runs.

**Hard constraints (inherited from Phase 1 spec §1):**
- 100% local execution. No cloud APIs.
- Hardware target unchanged: 16 GB VRAM, 64 GB RAM, Windows 11.
- Must reuse the local Qwen2.5-7B already on disk; only one new ML asset (VibeVoice-ASR-7B) to download.

---

## 2. Non-Goals

- Live (during-meeting) note-taking. Phase 1 already provides the live transcript stream; Phase 2 is strictly post-meeting.
- Edit/replay UI for the transcript. Notes are read-only output.
- Multi-language source; English only.
- Re-translation of Phase 1's Chinese subtitles. Phase 2 starts from `audio.wav` (the canonical record) and re-transcribes with VibeVoice for speaker labels — `transcript.txt` from Phase 1 is treated as a hint, not a source of truth.
- Speaker identification by name. We label speakers `Speaker 1`, `Speaker 2`, etc. — no voice fingerprinting against a roster.
- Cross-session knowledge linking, RAG, or vector search.
- Custom prompt templates beyond what ships in `summarizer.py`. Configurable, but not user-pluggable.

---

## 3. High-Level Architecture

```
sessions/<ts>/
├── audio.wav            ◄─── Phase 1 produces this
├── transcript.txt       ◄─── Phase 1 produces this (informational, not consumed)
│
│       Phase 2 entry: python -m meet_mirror_notes sessions/<ts>/
│
├── transcript_with_speakers.json   ◄─── stage 1 output (VibeVoice-ASR)
└── notes.md                        ◄─── stage 2 output (Qwen summarizer)
```

**Two stages, both batch / synchronous:**

1. **`vibevoice_asr.py`** — Load `microsoft/VibeVoice-ASR-7B`, batch-transcribe `audio.wav`, emit `transcript_with_speakers.json` with `[{ speaker, start_s, end_s, text }]` segments.
2. **`summarizer.py`** — Load Qwen2.5-7B (same Q4_K_M GGUF as Phase 1), feed the structured transcript through a meeting-notes prompt, emit `notes.md`.

Stages are independently re-runnable. If `transcript_with_speakers.json` already exists, stage 1 is skipped (cached). If `notes.md` exists, stage 2 is skipped unless `--force-summary`.

**No long-running processes, no Qt, no tray.** This is a one-shot CLI invocation.

---

## 4. Project Layout

```
src/meet_mirror_notes/                # Phase 2 package, separate from meet_mirror
├── __init__.py
├── __main__.py                       # entry: python -m meet_mirror_notes
├── cli.py                            # arg parsing
├── vibevoice_asr.py                  # stage 1
├── summarizer.py                     # stage 2
├── prompts.py                        # system + user prompt templates
└── types.py                          # TranscriptSegment dataclass

scripts/
└── notes_smoke.py                    # quick end-to-end test on a fixture
```

`pyproject.toml` adds a second console_scripts entry: `meet-mirror-notes = meet_mirror_notes.cli:main`.

---

## 5. Data Contracts (`src/meet_mirror_notes/types.py`)

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class TranscriptSegment:
    speaker: str            # "Speaker 1", "Speaker 2", ...
    start_s: float          # session-relative seconds
    end_s: float
    text: str               # English transcription, stripped
```

JSON schema for `transcript_with_speakers.json`:

```json
{
  "session_id": "2026-04-30_13-20-51",
  "audio_path": "audio.wav",
  "duration_s": 1834.2,
  "model": "microsoft/VibeVoice-ASR-7B",
  "language": "en",
  "segments": [
    { "speaker": "Speaker 1", "start_s": 0.4, "end_s": 5.7, "text": "..." },
    { "speaker": "Speaker 2", "start_s": 6.1, "end_s": 12.3, "text": "..." }
  ]
}
```

`notes.md` skeleton (rendered by `summarizer.py`):

```markdown
# 会议纪要 — 2026-04-30 13:20

## 概要
<3–5 sentence Chinese summary>

## 关键决策
- <bullet>
- <bullet>

## 行动项
- [ ] <action> (负责人:<speaker name or unknown>, 截止:<date if mentioned>)

## 未解决问题
- <bullet>

## 时间线
- 00:00 — 00:05 Speaker 1: <短中文复述>
- 00:06 — 00:12 Speaker 2: <短中文复述>
...
```

---

## 6. Components

### 6.1 VibeVoice ASR (`vibevoice_asr.py`)

- Model: `microsoft/VibeVoice-ASR-7B` via Hugging Face Transformers (loaded with `trust_remote_code=True` per the model card; we'll review the custom code before pinning a commit hash).
- Quantization: bitsandbytes 4-bit (`load_in_4bit=True`) is the first attempt. Estimated VRAM at Q4 is ~10–12 GB; combined with Whisper / Qwen *not* loaded (Phase 2 is its own process), this fits 16 GB comfortably.
- If 4-bit doesn't fit or quality regresses unacceptably, fall back to 8-bit with `device_map="auto"` so HF accelerate spills layers to CPU. The CLI exposes `--bits {4,8}` to choose.
- Pinned weight commit: TBD on first download; SHA256 of every `.safetensors` shard recorded in `BENCHMARKS.md`.
- Inference path:
  ```python
  output = model.generate(
      inputs=audio_inputs,
      task="transcribe",
      include_speaker_labels=True,   # VibeVoice-specific kwarg
      include_timestamps=True,
      max_new_tokens=4096,
  )
  ```
  Exact API depends on VibeVoice's chat template / generation interface; we will validate on first run and pin the call shape in this spec before merging the implementation.
- Audio chunking: VibeVoice-ASR-7B context is 30 s of audio per generate call. Long sessions are split into overlapping 30 s windows with 2 s overlap; segments at chunk boundaries are deduplicated by start_s.
- Output: list of `TranscriptSegment`, then serialized to JSON.

**Open question (must resolve before implementation):**
- VibeVoice's exact API for speaker-labelled output. Model card claims it; if the kwarg or output format differs from the assumption above, this section is rewritten.
- VRAM headroom under Q4: needs profiling on the target hardware. If we exceed 14 GB, we add layer offload as default rather than a fallback.

### 6.2 Summarizer (`summarizer.py`)

- Model: Qwen2.5-7B-Instruct Q4_K_M (the same GGUF Phase 1 already downloads to `models/`). No new dependency.
- Loaded with `llama-cpp-python` exactly like Phase 1's `TranslatorWorker`, but with a larger `n_ctx` (32 768 — a full hour-long meeting at ~150 tokens/min lands around 9 k tokens, with headroom for the prompt and the model's reasoning).
- System prompt focuses on:
  - Output language: Chinese only.
  - Style: 商务会议纪要 (professional meeting notes), avoid first-person, avoid filler.
  - Strict section structure (see §5 `notes.md` skeleton).
  - Speaker labels: keep `Speaker 1` / `Speaker 2` literal in 时间线 (timeline) but use `参会人员` / `提议人` etc. in prose where the speaker's role is clear from context.
  - Hallucination rule: if a piece of information is not in the transcript, write `不确定` rather than invent.
- Two-pass design for long meetings:
  - Pass 1 ("scope"): map-reduce. Segment transcript into ~10-min windows, ask Qwen for per-window mini-summaries with bullet points.
  - Pass 2 ("compose"): feed the concatenated mini-summaries plus the timeline back to Qwen with the final structure prompt → `notes.md`.
  - For meetings under 15 min, skip pass 1.
- Generation: `temperature=0.2`, `max_tokens=2048`, no streaming (write to disk on completion).
- Determinism caveats noted in the output: include the model+commit+config used as a frontmatter block in `notes.md` so notes from the same session are reproducible.

### 6.3 CLI (`cli.py` + `__main__.py`)

```bash
python -m meet_mirror_notes sessions/2026-04-30_13-20-51
```

Flags:
- `--bits {4,8}` — VibeVoice quantization (default 4)
- `--force-asr` — re-run stage 1 even if `transcript_with_speakers.json` exists
- `--force-summary` — re-run stage 2 even if `notes.md` exists
- `--asr-only` — stop after stage 1
- `--summary-only` — require existing `transcript_with_speakers.json`
- `--no-progress` — disable tqdm

Exit codes:
- `0` — success (or all stages skipped due to caches)
- `2` — invalid session directory (no audio.wav)
- `3` — stage 1 (ASR) failed
- `4` — stage 2 (summarizer) failed

Logs go to `sessions/<ts>/notes.log` and stderr (Loguru, INFO default).

---

## 7. Configuration

Phase 2 reuses `config.yaml` for translator settings (Qwen path) but adds a small section for the notes pipeline:

```yaml
notes:
  vibevoice_repo: "microsoft/VibeVoice-ASR-7B"
  vibevoice_bits: 4               # 4 or 8
  audio_chunk_s: 30               # window size for VibeVoice generate
  audio_overlap_s: 2              # dedupe window
  summary_n_ctx: 32768            # Qwen context for stage 2
  map_reduce_window_min: 10       # pass-1 window size; <15-min meetings skip pass 1
```

These fields are added to the existing Pydantic `Config` model, so a missing `notes:` key falls back to defaults and Phase 1 keeps working unchanged.

---

## 8. Error Handling

| Failure | Detection | Response |
|---|---|---|
| `audio.wav` missing | `FileNotFoundError` | Exit 2 with message |
| VibeVoice download fails | `huggingface_hub` exception | Exit 3, log full traceback, leave any partial file in HF cache for retry |
| VibeVoice OOM at 4-bit | `RuntimeError("CUDA out of memory")` | Auto-retry once at 8-bit; log decision |
| VibeVoice produces no speaker labels | empty `speaker` field on every segment | Fall back to `"Speaker 1"` for all and log warning; continue to summarizer (notes still useful without diarisation) |
| Qwen context overflow | input tokens > `n_ctx` | Force pass-1 map-reduce regardless of meeting length; if still overflowing, error with explicit token count |
| `notes.md` write fails | `OSError` | Exit 4, retain `transcript_with_speakers.json` for resume |

No automatic restart loops — Phase 2 is a CLI tool, the user re-runs.

---

## 9. Performance Budget

| Stage | Expected wall time (1 h meeting on RTX PRO 4000 16 GB) | Budget |
|---|---|---|
| VibeVoice load + Q4 init | 30 – 60 s | 90 s |
| VibeVoice transcription | 0.6× realtime → ~36 min for 60-min audio | 1× realtime |
| Qwen pass-1 (6 windows × ~300 ms each at the live numbers) | < 5 s | 30 s |
| Qwen pass-2 final compose | 5 – 15 s | 60 s |
| **Total for 1 h meeting** | **~37 min** | **~75 min** |

VRAM (Phase 2 only — Phase 1 not running):
- VibeVoice-ASR-7B Q4 + 30 s audio context: ~12 GB (estimated)
- Qwen2.5-7B Q4_K_M + 32 k context: ~6 GB
- They run in **separate stages**, not concurrently → peak VRAM = ~12 GB ✓

---

## 10. Testing

| Layer | Scope | Tooling |
|---|---|---|
| Unit | `types.py`, prompt builders, JSON schema serialisation, audio chunker | pytest, pure logic |
| Integration | VibeVoice on a 30-s English fixture, summarizer on a fixed transcript JSON | `@pytest.mark.gpu`, skipped in CI |
| End-to-end | `scripts/notes_smoke.py sessions/<fixture>/` → assert `notes.md` exists, has all 5 sections, length within bounds | manual |
| Quality | First three real sessions reviewed by hand against the source transcript | dogfood, no automation |

Fixtures: a 60-s 2-speaker English wav (recorded by user) + the matching expected `transcript_with_speakers.json` for regression testing. Lives under `tests/fixtures/notes/` (gitignored, user-provided).

---

## 11. Open Questions (must resolve before plan)

1. **VibeVoice-ASR-7B exact API.** Model card phrasing implies speaker labels and timestamps come from a single `generate` call but the kwarg names and output structure must be confirmed by loading the model on the target hardware. Spec §6.1 will be rewritten once verified.
2. **Q4 fit.** If 4-bit quant via bitsandbytes doesn't actually fit in 16 GB or produces unusable transcription quality, the fallback path (8-bit with offload) becomes the default. Need a 1-shot benchmark before drafting the plan.
3. **Speaker count assumption.** VibeVoice handles 2-speaker dialogues in its training; behaviour on 3+ speaker meetings is unverified. If diarisation accuracy collapses past 2 speakers, we may need to add a separate diarisation pass (pyannote) — out of scope unless this fails.
4. **Hour-long context for Qwen.** `n_ctx=32768` is plausible but the GGUF we're using was loaded with `n_ctx=8192` in Phase 1. We need to verify the model file supports the larger context (it does per Qwen's training, just not pre-allocated by Phase 1's runtime).
5. **Re-running cost.** VibeVoice at 0.6× realtime is fine for a one-off note-taking pass but costly to iterate on prompt changes. Consider caching `transcript_with_speakers.json` aggressively (already in design) and optionally exposing a `--summary-only` workflow.

---

## 12. Implementation Phasing (proposed for plan)

A vertical-slice plan analogous to Phase 1, smaller scope:

1. **Skeleton** — package layout, CLI parses args, loads `audio.wav`, prints "Notes pipeline ready".
2. **Stage 1 (VibeVoice)** — model load, single 30-s chunk transcription with speaker labels, write JSON. No chunking yet.
3. **Audio chunking + dedupe** — window the full file, dedupe overlap, full-session JSON output.
4. **Stage 2 (Qwen summarizer)** — single-pass summarization for short meetings (<15 min), produce `notes.md` with all 5 sections.
5. **Map-reduce** — pass-1 mini-summaries for long meetings, pass-2 compose.
6. **CLI ergonomics** — caches, force flags, exit codes, logging to `notes.log`.
7. **Hardening + benchmarks** — error matrix from §8, append `BENCHMARKS.md` Phase 2 section with first real measurements.

Each slice ends with a runnable system. Detailed plan to follow once §11 open questions are resolved on the target machine.

---

## 13. Out of Scope (deferred to a hypothetical Phase 3)

- Speaker identification (matching `Speaker 1` to a real name).
- Multi-language source meetings (only ja/zh/etc.).
- Action-item export to ticketing systems (Jira, Linear).
- Anonymisation / redaction of transcripts before notes are stored.
- A web/Obsidian viewer for the produced notes.
- Re-summarisation across multiple sessions (weekly digest).
