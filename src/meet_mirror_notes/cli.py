from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]


EXIT_OK = 0
EXIT_BAD_SESSION = 2
EXIT_ASR_FAILED = 3
EXIT_SUMMARY_FAILED = 4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="meet-mirror-notes",
        description="Post-meeting notes pipeline: VibeVoice ASR + Qwen summarizer",
    )
    parser.add_argument(
        "session_dir",
        type=Path,
        help="Path to a Phase 1 session directory containing audio.wav",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="VibeVoice quantization (default 4)",
    )
    parser.add_argument(
        "--force-asr",
        action="store_true",
        help="Re-run stage 1 even if transcript_with_speakers.json exists",
    )
    parser.add_argument(
        "--force-summary",
        action="store_true",
        help="Re-run stage 2 even if notes.md exists",
    )
    parser.add_argument(
        "--asr-only",
        action="store_true",
        help="Stop after stage 1; do not run summarizer",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip stage 1; require existing transcript_with_speakers.json",
    )
    parser.add_argument(
        "--speakers",
        type=int,
        default=2,
        help="Number of speakers for diarization (default 2)",
    )
    parser.add_argument(
        "--whisper-model",
        default="large-v3-turbo",
        help="faster-whisper model name (default large-v3-turbo)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device for Whisper (default cuda)",
    )
    parser.add_argument(
        "--compute-type",
        default="float16",
        choices=["float16", "int8", "int8_float16", "float32"],
        help="Whisper compute type (default float16)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    return parser.parse_args(argv)


def configure_logging(session_dir: Path) -> None:
    log_path = session_dir / "notes.log"
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(
        log_path,
        level="DEBUG",
        rotation="10 MB",
        retention=3,
        enqueue=True,
        encoding="utf-8",
    )


def validate_session(session_dir: Path) -> int | None:
    if not session_dir.exists():
        print(f"Session dir not found: {session_dir}", file=sys.stderr)
        return EXIT_BAD_SESSION
    if not session_dir.is_dir():
        print(f"Not a directory: {session_dir}", file=sys.stderr)
        return EXIT_BAD_SESSION
    audio = session_dir / "audio.wav"
    if not audio.exists():
        print(f"No audio.wav in {session_dir}", file=sys.stderr)
        return EXIT_BAD_SESSION
    return None


def audio_duration_s(audio_path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(audio_path))
    return float(info.duration)


def run_stage1_asr_diarize(
    session_dir: Path, audio_path: Path, duration: float, args: argparse.Namespace
) -> int | None:
    """Stage 1: transcribe + diarize. Returns exit code on error, else None."""
    transcript_json = session_dir / "transcript_with_speakers.json"
    if transcript_json.exists() and not args.force_asr:
        logger.info(f"[skip stage 1] {transcript_json.name} exists "
                    f"(use --force-asr to re-run)")
        return None

    try:
        from .asr_diarize import transcribe_and_diarize, write_transcript_json
    except ImportError as e:
        logger.error(f"Stage 1 deps missing: {e}")
        return EXIT_ASR_FAILED

    try:
        segments = transcribe_and_diarize(
            audio_path,
            n_speakers=args.speakers,
            model_name=args.whisper_model,
            device=args.device,
            compute_type=args.compute_type,
        )
    except Exception as e:
        logger.exception(f"Stage 1 failed: {e}")
        return EXIT_ASR_FAILED

    write_transcript_json(
        transcript_json, segments, audio_path, duration,
        args.whisper_model, args.speakers,
    )
    logger.info(f"Stage 1: {len(segments)} segments across "
                f"{args.speakers} speakers")
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    session_dir = args.session_dir.resolve()

    err = validate_session(session_dir)
    if err is not None:
        return err

    configure_logging(session_dir)
    audio_path = session_dir / "audio.wav"
    duration = audio_duration_s(audio_path)
    logger.info(f"Session: {session_dir}")
    logger.info(f"audio.wav duration: {duration:.1f}s ({duration / 60:.1f} min)")
    logger.info(f"speakers={args.speakers} force_asr={args.force_asr} "
                f"force_summary={args.force_summary} "
                f"asr_only={args.asr_only} summary_only={args.summary_only}")

    if not args.summary_only:
        err = run_stage1_asr_diarize(session_dir, audio_path, duration, args)
        if err is not None:
            return err

    if args.asr_only:
        logger.info("--asr-only: stopping after stage 1")
        return EXIT_OK

    transcript_json = session_dir / "transcript_with_speakers.json"
    if not transcript_json.exists():
        logger.error("No transcript_with_speakers.json; run without --summary-only first")
        return EXIT_BAD_SESSION

    logger.info("Stage 2 (Qwen summarizer) lands in Slice 4.")
    print("Stage 1 complete; stage 2 pending.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
