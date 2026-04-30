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

from .prompts import (
    COMPOSE_USER_TEMPLATE,
    SINGLE_PASS_USER_TEMPLATE,
    SUMMARY_SYSTEM_PROMPT,
    WINDOW_SUMMARY_USER_TEMPLATE,
)

# Tunables
WINDOW_S = 600                    # 10 min per pass-1 window
ROUTE_RESERVE_TOKENS = 3000       # leave room in n_ctx for prompt + response
TOKENS_PER_CHAR = 1 / 3           # rough estimate for EN + timestamps + speaker labels


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


def _estimate_tokens(text: str) -> int:
    return int(len(text) * TOKENS_PER_CHAR)


def _group_segments_into_windows(
    segments: list[dict], window_s: float = WINDOW_S
) -> list[list[dict]]:
    """Bucket segments into ≈ window_s slots based on start_s."""
    if not segments:
        return []
    windows: list[list[dict]] = []
    current: list[dict] = []
    window_start = segments[0]["start_s"]
    for s in segments:
        if s["start_s"] - window_start >= window_s and current:
            windows.append(current)
            current = []
            window_start = s["start_s"]
        current.append(s)
    if current:
        windows.append(current)
    return windows


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


def summarize_map_reduce(
    transcript_segments: list[dict],
    model_path: Path,
    n_ctx: int = 16384,
    window_s: float = WINDOW_S,
    pass1_max_tokens: int = 512,
    pass2_max_tokens: int = 2048,
    temperature: float = 0.2,
) -> str:
    """Two-pass summarization for transcripts that don't fit in n_ctx.

    Pass 1 (map): split the transcript into ~window_s slices, ask Qwen
    for 3–6 short bullet points per slice.
    Pass 2 (reduce): feed all bullets + the full timeline back through
    the spec §6.2 system prompt to compose the final notes.md body.
    """
    windows = _group_segments_into_windows(transcript_segments, window_s)
    logger.info(
        f"Map-reduce: {len(windows)} windows of ~{int(window_s / 60)} min "
        f"({len(transcript_segments)} segments total)"
    )
    llm = load_qwen(model_path, n_ctx=n_ctx)

    # --- Pass 1: per-window mini-summaries ---
    window_summaries: list[str] = []
    t_pass1 = time.perf_counter()
    for i, w in enumerate(windows):
        start_min = int(w[0]["start_s"] / 60)
        end_min = int(w[-1]["end_s"] / 60)
        chunk = transcript_as_prompt(w)
        user_msg = WINDOW_SUMMARY_USER_TEMPLATE.format(
            start_min=start_min, end_min=end_min, transcript=chunk
        )
        t0 = time.perf_counter()
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=temperature,
            max_tokens=pass1_max_tokens,
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        body = resp["choices"][0]["message"]["content"].strip()
        logger.info(
            f"  pass-1 window {i + 1}/{len(windows)} "
            f"({start_min}–{end_min} min) done in {elapsed_ms}ms"
        )
        window_summaries.append(body)
    logger.info(
        f"Pass 1 total: {int((time.perf_counter() - t_pass1) * 1000)}ms"
    )

    # --- Pass 2: compose final notes ---
    bullets_text = "\n\n".join(
        f"### 第 {i + 1} 段 ({int(w[0]['start_s'] / 60)}–"
        f"{int(w[-1]['end_s'] / 60)} 分钟)\n{summary}"
        for i, (w, summary) in enumerate(
            zip(windows, window_summaries, strict=True)
        )
    )
    timeline = transcript_as_prompt(transcript_segments)

    # If the combined pass-2 input still overflows n_ctx, drop the raw
    # timeline from the prompt and append it ourselves below the LLM's
    # output. Bullets alone are enough for the LLM to compose §概要 / 关键决策
    # / 行动项 / 未解决问题 reliably.
    drop_timeline = False
    if _estimate_tokens(bullets_text) + _estimate_tokens(timeline) + ROUTE_RESERVE_TOKENS > n_ctx:
        logger.warning(
            "Pass-2 input would overflow n_ctx; dropping raw timeline. "
            "Notes.md timeline will be appended programmatically from segments."
        )
        timeline = "(完整时间线略 — 由代码从 transcript_with_speakers.json 直接附加)"
        drop_timeline = True

    user_msg = COMPOSE_USER_TEMPLATE.format(
        window_summaries=bullets_text, timeline=timeline
    )
    t0 = time.perf_counter()
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
        max_tokens=pass2_max_tokens,
    )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(f"Pass 2 (compose) done in {elapsed_ms}ms")
    body = resp["choices"][0]["message"]["content"].strip()

    if drop_timeline:
        # Append the full timeline ourselves so the section isn't lost.
        body += "\n\n## 完整时间线（自动附加）\n\n"
        body += "\n".join(
            f"- [{_format_ts(s['start_s'])}] {s['speaker']}: {s['text']}"
            for s in transcript_segments
        )

    return body


def summarize_auto(
    transcript_segments: list[dict],
    model_path: Path,
    n_ctx: int = 16384,
    max_tokens: int = 2048,
    temperature: float = 0.2,
) -> str:
    """Pick single-pass vs map-reduce by transcript size."""
    transcript_text = transcript_as_prompt(transcript_segments)
    est = _estimate_tokens(transcript_text)
    if est + ROUTE_RESERVE_TOKENS <= n_ctx:
        logger.info(
            f"Routing: single-pass (≈{est} tokens transcript, n_ctx={n_ctx})"
        )
        return summarize_single_pass(
            transcript_segments,
            model_path,
            n_ctx=n_ctx,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    logger.info(
        f"Routing: map-reduce (≈{est} tokens transcript exceeds "
        f"n_ctx={n_ctx} - {ROUTE_RESERVE_TOKENS} reserve)"
    )
    return summarize_map_reduce(
        transcript_segments,
        model_path,
        n_ctx=n_ctx,
        pass2_max_tokens=max_tokens,
        temperature=temperature,
    )


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
    n_ctx: int = 16384,
) -> Path:
    """Orchestrator: read JSON, summarize, write notes.md. Returns the
    notes.md path. Raises on error."""
    transcript_path = session_dir / "transcript_with_speakers.json"
    notes_path = session_dir / "notes.md"

    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    segments = payload["segments"]

    body = summarize_auto(segments, qwen_model_path, n_ctx=n_ctx)
    write_notes_md(
        notes_path,
        body,
        transcript_path,
        payload.get("duration_s", 0.0),
        qwen_model_path,
        payload.get("n_speakers", 1),
    )
    return notes_path
