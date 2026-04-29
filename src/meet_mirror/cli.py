from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from .config import Config


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

    print("Meet Mirror ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
