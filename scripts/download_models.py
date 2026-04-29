"""Download model weights into ./models/.

Slice 0: stub. Prints required artefacts and exits.
Slice 2 fills in faster-whisper auto-download via WhisperModel cache.
Slice 3 fills in Qwen GGUF download from HuggingFace + sha256 verification.
"""

from __future__ import annotations

import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

REQUIRED = [
    {
        "name": "faster-whisper large-v3-turbo",
        "purpose": "Real-time English speech-to-text (Slice 2).",
        "how_to_get": (
            "Auto-downloads on first WhisperModel('large-v3-turbo') call into "
            "the HuggingFace cache (~/.cache/huggingface/hub). No manual step."
        ),
        "size_gb": 1.6,
    },
    {
        "name": "Qwen2.5-7B-Instruct-Q4_K_M GGUF",
        "purpose": "Local English-to-Chinese translator (Slice 3).",
        "how_to_get": (
            "Download from "
            "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/"
            "qwen2-5-7b-instruct-q4_k_m.gguf "
            "into models/qwen2.5-7b-instruct-q4_k_m.gguf "
            "(automated in Slice 3 with sha256 verification)."
        ),
        "size_gb": 4.4,
    },
]


def main() -> int:
    print(f"Models directory: {MODELS_DIR}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print()
    print("Required model artefacts:")
    print()
    for item in REQUIRED:
        print(f"- {item['name']} (~{item['size_gb']:.1f} GB)")
        print(f"    Purpose: {item['purpose']}")
        print(f"    How:     {item['how_to_get']}")
        print()
    print("This script is a stub in Slice 0; actual downloads land in Slice 2 and Slice 3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
