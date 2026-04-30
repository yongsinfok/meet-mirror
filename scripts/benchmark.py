from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def benchmark_whisper(wav_path: Path, model_name: str, device: str, compute_type: str) -> None:
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        print(f"WARNING: sample rate {sr} != 16000; transcription may be inaccurate")
    duration = len(audio) / sr
    print(f"Audio: {wav_path.name}  duration={duration:.2f}s  sr={sr}")

    from faster_whisper import WhisperModel

    print(f"Loading Whisper '{model_name}' on {device}/{compute_type}...")
    t0 = time.perf_counter()
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"Model load: {load_ms:.0f}ms")

    # Warmup (compile kernels, prime caches)
    print("Warmup pass (1s)...")
    warmup_audio = audio[: min(16000, len(audio))].astype(np.float32)
    list(
        model.transcribe(
            warmup_audio, language="en", vad_filter=False, beam_size=1
        )[0]
    )

    print(f"Transcribing {duration:.2f}s ...")
    t0 = time.perf_counter()
    segments, _info = model.transcribe(
        audio,
        language="en",
        vad_filter=False,
        beam_size=1,
        condition_on_previous_text=True,
    )
    seg_list = list(segments)
    elapsed = time.perf_counter() - t0

    text = " ".join(s.text.strip() for s in seg_list).strip()
    n_tokens = sum(len(s.tokens) for s in seg_list)

    print()
    print(f"Wall time:   {elapsed * 1000:.0f}ms")
    print(f"RTF:         {elapsed / duration:.3f}x  (lower is better; <1 = faster than realtime)")
    print(f"Tokens:      {n_tokens}")
    print(f"Tokens/sec:  {n_tokens / elapsed:.1f}")
    print()
    print(f"Text: {text[:300]}{'...' if len(text) > 300 else ''}")


def main() -> int:
    parser = argparse.ArgumentParser(prog="benchmark", description="Whisper transcription benchmark")
    parser.add_argument("wav", type=Path, help="Input wav (any sample rate; mono preferred)")
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--compute-type",
        default="float16",
        choices=["float16", "int8", "int8_float16", "float32"],
    )
    args = parser.parse_args()

    if not args.wav.exists():
        print(f"File not found: {args.wav}", file=sys.stderr)
        return 2

    benchmark_whisper(args.wav, args.model, args.device, args.compute_type)
    return 0


if __name__ == "__main__":
    sys.exit(main())
