"""Phase 2 Slice 4 — Qwen2.5-7B summarizer (single-pass).

Reads transcript_with_speakers.json, feeds the transcript to the
shared Qwen GGUF (the Phase 1 download), produces notes.md per the
spec §6.2 structure.

Map-reduce for hour-long meetings is a follow-up slice; this module
handles the short-meeting path (transcript fits in 8k context).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from loguru import logger

from .prompts import SINGLE_PASS_USER_TEMPLATE, SUMMARY_SYSTEM_PROMPT


def _format_ts(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def transcript_as_prompt(segments: list[dict]) -> str:
    """Render the transcript into a flat text the LLM can read."""
    lines = []
    for s in segments:
        ts = _format_ts(s["start_s"])
        lines.append(f"[{ts}] {s['speaker']}: {s['text']}")
    return "\n".join(lines)


def load_qwen(model_path: Path, n_ctx: int = 8192) -> object:
    from llama_cpp import Llama

    logger.info(f"Loading Qwen summarizer from {model_path} (n_ctx={n_ctx})")
    return Llama(
        model_path=str(model_path),
        n_gpu_layers=-1,
        n_ctx=n_ctx,
        flash_attn=True,
        verbose=False,
    )


def summarize_single_pass(
    transcript_segments: list[dict],
    model_path: Path,
    n_ctx: int = 8192,
    max_tokens: int = 2048,
    temperature: float = 0.2,
) -> str:
    """One Qwen call — for transcripts that fit in n_ctx."""
    llm = load_qwen(model_path, n_ctx=n_ctx)
    transcript_text = transcript_as_prompt(transcript_segments)
    user_msg = SINGLE_PASS_USER_TEMPLATE.format(transcript=transcript_text)

    logger.info(
        f"Summarizing {len(transcript_segments)} segments "
        f"({len(transcript_text)} chars of transcript)"
    )
    t0 = time.perf_counter()
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(f"Summarization complete in {elapsed_ms}ms")
    return resp["choices"][0]["message"]["content"].strip()


def write_notes_md(
    out_path: Path,
    notes_body: str,
    transcript_path: Path,
    duration_s: float,
    model_path: Path,
    n_speakers: int,
) -> None:
    """Write notes.md with a small frontmatter block for reproducibility."""
    frontmatter = (
        "---\n"
        f"session: {transcript_path.parent.name}\n"
        f"duration_s: {duration_s:.1f}\n"
        f"diarizer: resemblyzer+sklearn\n"
        f"n_speakers: {n_speakers}\n"
        f"summarizer_model: {model_path.name}\n"
        "---\n\n"
    )
    out_path.write_text(frontmatter + notes_body + "\n", encoding="utf-8")
    logger.info(f"Wrote {out_path}")


def run_stage2_summarize(
    session_dir: Path,
    qwen_model_path: Path,
    n_ctx: int = 8192,
) -> Path:
    """Orchestrator: read JSON, summarize, write notes.md. Returns the
    notes.md path. Raises on error."""
    transcript_path = session_dir / "transcript_with_speakers.json"
    notes_path = session_dir / "notes.md"

    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    segments = payload["segments"]

    body = summarize_single_pass(segments, qwen_model_path, n_ctx=n_ctx)
    write_notes_md(
        notes_path,
        body,
        transcript_path,
        payload.get("duration_s", 0.0),
        qwen_model_path,
        payload.get("n_speakers", 1),
    )
    return notes_path
