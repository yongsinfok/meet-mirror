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
    logger.info(f"bits={args.bits} force_asr={args.force_asr} "
                f"force_summary={args.force_summary} "
                f"asr_only={args.asr_only} summary_only={args.summary_only}")

    print("Notes pipeline ready (skeleton — Slice 2 wires up VibeVoice ASR)")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
