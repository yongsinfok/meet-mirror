"""Download model weights into ./models/.

- Whisper large-v3-turbo: auto-downloads to the HuggingFace cache on first
  WhisperModel(...) call. We just print the trigger.
- Qwen2.5-7B-Instruct Q4_K_M GGUF: pulled here into ./models/ with SHA256
  logged for identity verification.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

QWEN_REPO = "Qwen/Qwen2.5-7B-Instruct-GGUF"
QWEN_FILES = [
    "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
    "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
]
# llama.cpp loads the whole split set when pointed at the first shard.
QWEN_PRIMARY = QWEN_FILES[0]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download_qwen() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    primary: Path | None = None
    for fname in QWEN_FILES:
        target = MODELS_DIR / fname
        if target.exists():
            size_gb = target.stat().st_size / 1e9
            print(f"[skip] {target.name} already present ({size_gb:.2f} GB)")
            print(f"  sha256: {sha256_file(target)}")
        else:
            print(f"Downloading {QWEN_REPO} :: {fname} ...")
            cached = hf_hub_download(
                repo_id=QWEN_REPO,
                filename=fname,
                local_dir=str(MODELS_DIR),
            )
            p = Path(cached)
            size_gb = p.stat().st_size / 1e9
            print(f"  Saved: {p} ({size_gb:.2f} GB)")
            print(f"  sha256: {sha256_file(p)}")
            target = p
        if fname == QWEN_PRIMARY:
            primary = target

    assert primary is not None
    return primary


def whisper_status() -> None:
    print("Whisper large-v3-turbo:")
    print("  - Auto-downloads to HuggingFace cache on first WhisperModel(...) call.")
    print("  - Repo: mobiuslabsgmbh/faster-whisper-large-v3-turbo")
    print("  - Trigger: `python main.py --mode console`")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(prog="download_models")
    parser.add_argument(
        "--whisper-only",
        action="store_true",
        help="Print Whisper info, skip Qwen download",
    )
    parser.add_argument(
        "--qwen-only",
        action="store_true",
        help="Skip Whisper info, download Qwen only",
    )
    args = parser.parse_args()

    print(f"Models directory: {MODELS_DIR}")
    print()

    if not args.qwen_only:
        whisper_status()

    if not args.whisper_only:
        download_qwen()

    return 0


if __name__ == "__main__":
    sys.exit(main())
