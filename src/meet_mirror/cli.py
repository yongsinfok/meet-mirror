from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

from loguru import logger

from .config import Config

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]


def configure_logging(cfg: Config) -> None:
    log_path = Path(cfg.logging.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level=cfg.logging.level)
    logger.add(
        log_path,
        level="DEBUG",
        rotation=f"{cfg.logging.rotation_mb} MB",
        retention=5,
        enqueue=True,
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="meet-mirror")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: ./config.yaml)",
    )
    parser.add_argument(
        "--mode",
        choices=["gui", "console"],
        default="gui",
        help="gui: floating subtitle bar (default). console: print to stdout (debug)",
    )
    return parser.parse_args(argv)


def _format_relative(t_abs: float, session_start: float) -> str:
    delta = max(0.0, t_abs - session_start)
    h = int(delta // 3600)
    m = int((delta % 3600) // 60)
    s = int(delta % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def run_console(cfg: Config) -> int:
    from .pipeline import Pipeline

    pipeline = Pipeline(cfg)
    stop_consumer = threading.Event()
    session_start = time.time()

    def consume() -> None:
        while not stop_consumer.is_set():
            try:
                seg = pipeline.subtitle_q.get(timeout=0.5)
            except queue.Empty:
                continue
            ts = _format_relative(seg.audio_ts_start, session_start)
            print(f"[{ts}] EN: {seg.en_text}", flush=True)
            print(
                f"           ZH: {seg.zh_text} (lat={seg.translation_latency_ms}ms)",
                flush=True,
            )

    consumer = threading.Thread(target=consume, name="ConsoleConsumer", daemon=True)

    pipeline.start()
    consumer.start()
    print("Meet Mirror ready (console mode). Ctrl+C to exit.", flush=True)

    try:
        while not pipeline.stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nShutting down...", flush=True)
    finally:
        pipeline.stop()
        stop_consumer.set()
        consumer.join(timeout=2.0)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return 2

    cfg = Config.load(config_path)
    configure_logging(cfg)
    logger.info("Loaded config from {}", config_path)
    logger.info("Mode: {}", args.mode)

    if args.mode == "console":
        return run_console(cfg)

    # gui mode: floating subtitle UI lands in Slice 4
    print("Meet Mirror ready")
    print("(GUI mode pending Slice 4 — use --mode console for live transcription.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
