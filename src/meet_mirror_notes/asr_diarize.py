"""Phase 2 Slice 2 — ASR + speaker diarization.

Combines faster-whisper word-level transcription with Resemblyzer
voice embeddings + agglomerative clustering to produce
[{speaker, start_s, end_s, text}] segments. Per spec §6.1 (revised
2026-04-30 to drop VibeVoice in favour of this open pipeline).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger

from .types import TranscriptSegment

SAMPLE_RATE = 16000  # all Phase 1 audio.wav is captured at this rate
INTRA_TURN_PAUSE_S = 0.3   # gaps shorter than this are mid-utterance
INTER_TURN_PAUSE_S = 0.8   # gaps this long start a new utterance
MAX_UTTERANCE_S = 15.0     # force-split utterances longer than this
SHORT_UTTERANCE_S = 1.0    # below this we inherit the previous speaker
RESEMBLYZER_MIN_S = 1.6    # Resemblyzer needs ~1.6s of audio per embedding


@dataclass
class WordToken:
    word: str
    start_s: float
    end_s: float


def _hf_offline_if_cached() -> None:
    """Reuse Phase 1's heuristic so corp networks don't stall the load."""
    if "HF_HUB_OFFLINE" in os.environ:
        return
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache_dir.exists():
        return
    for p in cache_dir.iterdir():
        if not p.is_dir() or "whisper" not in p.name.lower():
            continue
        snapshots = p / "snapshots"
        if snapshots.exists() and any(snapshots.iterdir()):
            os.environ["HF_HUB_OFFLINE"] = "1"
            logger.info("Whisper cache hit; setting HF_HUB_OFFLINE=1")
            return


def transcribe_words(
    audio_path: Path,
    model_name: str = "large-v3-turbo",
    device: str = "cuda",
    compute_type: str = "float16",
) -> list[WordToken]:
    """Run faster-whisper with word_timestamps=True over the full file."""
    _hf_offline_if_cached()
    from faster_whisper import WhisperModel

    logger.info(f"Loading Whisper {model_name} ({device}/{compute_type}) ...")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    logger.info(f"Transcribing {audio_path.name} ...")
    segments, info = model.transcribe(
        str(audio_path),
        language="en",
        vad_filter=False,
        beam_size=1,
        condition_on_previous_text=True,
        word_timestamps=True,
    )

    words: list[WordToken] = []
    for seg in segments:
        if seg.words is None:
            # Rare: whisper produced a segment with no word timestamps.
            words.append(WordToken(seg.text.strip(), float(seg.start), float(seg.end)))
            continue
        for w in seg.words:
            t = (w.word or "").strip()
            if not t:
                continue
            words.append(WordToken(t, float(w.start), float(w.end)))

    logger.info(f"Whisper produced {len(words)} word tokens "
                f"over {info.duration:.1f}s of audio")
    return words


def group_words_into_utterances(
    words: list[WordToken],
    inter_pause_s: float = INTER_TURN_PAUSE_S,
    max_duration_s: float = MAX_UTTERANCE_S,
) -> list[TranscriptSegment]:
    """Greedy chunker.

    Splits on:
      1. silence ≥ inter_pause_s (turn boundary), OR
      2. accumulated duration ≥ max_duration_s (force split — prefer the
         longest gap inside the window if any > 0.2s, else cut at the
         current word).

    Continuous-speech sources (lectures, narration) hit rule 2; natural
    conversation hits rule 1.
    """
    if not words:
        return []

    out: list[TranscriptSegment] = []
    cur: list[WordToken] = [words[0]]

    def flush() -> None:
        out.append(TranscriptSegment(
            speaker="Speaker 0",  # placeholder; clustering replaces
            start_s=cur[0].start_s,
            end_s=cur[-1].end_s,
            text=" ".join(w.word for w in cur).strip(),
        ))

    for w in words[1:]:
        gap = w.start_s - cur[-1].end_s
        cur_duration = cur[-1].end_s - cur[0].start_s

        if gap >= inter_pause_s:
            flush()
            cur = [w]
            continue

        if cur_duration + (w.end_s - cur[-1].end_s) >= max_duration_s:
            # Try to split at the largest internal gap (≥ 0.2s); else
            # cut here at the current word.
            best_idx = -1
            best_gap = 0.2
            for i in range(1, len(cur)):
                g = cur[i].start_s - cur[i - 1].end_s
                if g > best_gap:
                    best_gap = g
                    best_idx = i
            if best_idx > 0:
                head, tail = cur[:best_idx], cur[best_idx:]
                cur = head
                flush()
                cur = [*tail, w]
            else:
                flush()
                cur = [w]
            continue

        cur.append(w)

    flush()
    logger.info(f"Grouped into {len(out)} utterances")
    return out


def _embed_segments(
    audio: np.ndarray,
    sr: int,
    segments: list[TranscriptSegment],
) -> tuple[np.ndarray, list[bool]]:
    """Returns (embeddings_matrix, is_too_short_flags) of shape
    (N, 256) and (N,). Too-short utterances get a zero embedding and
    will inherit the prev-speaker downstream."""
    from resemblyzer import VoiceEncoder, preprocess_wav

    encoder = VoiceEncoder(verbose=False)
    embeddings: list[np.ndarray] = []
    too_short: list[bool] = []
    for s in segments:
        dur = s.end_s - s.start_s
        if dur < RESEMBLYZER_MIN_S:
            embeddings.append(np.zeros(256, dtype=np.float32))
            too_short.append(True)
            continue
        i0 = int(s.start_s * sr)
        i1 = int(s.end_s * sr)
        clip = audio[i0:i1]
        wav = preprocess_wav(clip, source_sr=sr)
        emb = encoder.embed_utterance(wav)
        embeddings.append(emb.astype(np.float32))
        too_short.append(False)
    return np.stack(embeddings), too_short


def _cluster(embeddings: np.ndarray, n_speakers: int) -> np.ndarray:
    """Agglomerative clustering with cosine distance."""
    from sklearn.cluster import AgglomerativeClustering

    if len(embeddings) <= n_speakers:
        # Trivial: every segment is its own speaker, label by order.
        return np.arange(len(embeddings))
    clusterer = AgglomerativeClustering(
        n_clusters=n_speakers,
        metric="cosine",
        linkage="average",
    )
    return clusterer.fit_predict(embeddings)


def _relabel_by_first_appearance(cluster_ids: np.ndarray) -> dict[int, str]:
    """Map raw cluster ids to 'Speaker 1', 'Speaker 2', ... in the order
    each id first appears in the timeline."""
    seen: list[int] = []
    for cid in cluster_ids:
        if cid not in seen:
            seen.append(int(cid))
    return {cid: f"Speaker {i + 1}" for i, cid in enumerate(seen)}


def diarize(
    segments: list[TranscriptSegment],
    audio: np.ndarray,
    sr: int,
    n_speakers: int,
) -> list[TranscriptSegment]:
    """Replace each segment.speaker with a clustered speaker label."""
    if not segments:
        return []
    if n_speakers <= 1:
        return [TranscriptSegment("Speaker 1", s.start_s, s.end_s, s.text)
                for s in segments]

    embeddings, too_short = _embed_segments(audio, sr, segments)

    # Cluster only the non-short segments to avoid junk-embedding noise.
    long_idx = [i for i, ts in enumerate(too_short) if not ts]
    if len(long_idx) < n_speakers:
        # Not enough material to cluster reliably — fall back to single speaker.
        logger.warning(
            f"Only {len(long_idx)} utterances ≥ {RESEMBLYZER_MIN_S}s; "
            f"insufficient for {n_speakers}-way clustering. Labelling all "
            f"as Speaker 1."
        )
        return [TranscriptSegment("Speaker 1", s.start_s, s.end_s, s.text)
                for s in segments]

    long_embeddings = embeddings[long_idx]
    long_clusters = _cluster(long_embeddings, n_speakers)

    # Stitch back: short utterances inherit from the previous long one.
    cluster_ids = np.full(len(segments), -1, dtype=int)
    for i, lid in zip(long_idx, long_clusters, strict=True):
        cluster_ids[i] = lid
    last = 0
    for i in range(len(cluster_ids)):
        if cluster_ids[i] == -1:
            cluster_ids[i] = last
        else:
            last = cluster_ids[i]

    label_for = _relabel_by_first_appearance(cluster_ids)
    return [
        TranscriptSegment(
            speaker=label_for[int(cid)],
            start_s=s.start_s,
            end_s=s.end_s,
            text=s.text,
        )
        for s, cid in zip(segments, cluster_ids, strict=True)
    ]


def transcribe_and_diarize(
    audio_path: Path,
    n_speakers: int,
    model_name: str = "large-v3-turbo",
    device: str = "cuda",
    compute_type: str = "float16",
) -> list[TranscriptSegment]:
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        logger.warning(
            f"audio.wav sample rate {sr} != {SAMPLE_RATE}; "
            "Whisper handles internally but Resemblyzer prefers 16kHz"
        )

    words = transcribe_words(audio_path, model_name, device, compute_type)
    if not words:
        return []
    segments = group_words_into_utterances(words)
    diarized = diarize(segments, audio, sr, n_speakers)
    return diarized


def write_transcript_json(
    out_path: Path,
    segments: list[TranscriptSegment],
    audio_path: Path,
    duration_s: float,
    model_name: str,
    n_speakers: int,
) -> None:
    payload = {
        "session_id": audio_path.parent.name,
        "audio_path": audio_path.name,
        "duration_s": duration_s,
        "model": model_name,
        "diarizer": "resemblyzer+sklearn-agglomerative",
        "n_speakers": n_speakers,
        "language": "en",
        "segments": [
            {
                "speaker": s.speaker,
                "start_s": round(s.start_s, 3),
                "end_s": round(s.end_s, 3),
                "text": s.text,
            }
            for s in segments
        ],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    logger.info(f"Wrote {out_path}")
