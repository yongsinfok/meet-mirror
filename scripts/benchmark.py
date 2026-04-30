from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]


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


BUSINESS_SENTENCES = [
    "We need to align on the timeline before the demo.",
    "Salesforce integration is blocked on the API rate limit fix.",
    "John will follow up with the customer next Monday.",
    "Can you share the latest project plan in the channel?",
    "The migration to AWS should be done by the end of Q3.",
    "Let's schedule a sync with the engineering team this week.",
    "We are seeing significant churn in the enterprise segment.",
    "The deal closed yesterday for about two million dollars.",
    "Please review the proposal before tomorrow's stand-up.",
    "We've decided to deprecate the legacy authentication module.",
]


def benchmark_translator(model_path: Path, sentences: list[str]) -> None:
    from meet_mirror.translator import SYSTEM_PROMPT

    if not model_path.exists():
        print(f"Model not found: {model_path}", file=sys.stderr)
        print("Run: python scripts/download_models.py", file=sys.stderr)
        sys.exit(2)

    print(f"Loading Qwen GGUF: {model_path}")
    from llama_cpp import Llama

    t0 = time.perf_counter()
    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=-1,
        n_ctx=8192,
        flash_attn=True,
        verbose=False,
    )
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"Model load: {load_ms:.0f}ms")

    print("Warmup pass...")
    llm.create_chat_completion(
        messages=[{"role": "user", "content": "hi"}], max_tokens=10
    )

    latencies: list[float] = []
    print()
    for s in sentences:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": s},
        ]
        t0 = time.perf_counter()
        resp = llm.create_chat_completion(
            messages=msgs, temperature=0.2, max_tokens=300, stop=["\n\n"]
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)
        zh = resp["choices"][0]["message"]["content"].strip()
        print(f"  [{elapsed_ms:5.0f}ms] EN: {s}")
        print(f"            ZH: {zh}")

    arr = sorted(latencies)
    p50 = arr[len(arr) // 2]
    p95 = arr[min(len(arr) - 1, int(round(len(arr) * 0.95) - 1))]
    mean = sum(arr) / len(arr)
    print()
    print(f"Translation latency over {len(arr)} sentences:")
    print(f"  P50:  {p50:.0f} ms")
    print(f"  P95:  {p95:.0f} ms")
    print(f"  Mean: {mean:.0f} ms")
    print(f"  Min:  {arr[0]:.0f} ms")
    print(f"  Max:  {arr[-1]:.0f} ms")


def main() -> int:
    parser = argparse.ArgumentParser(prog="benchmark", description="Meet Mirror benchmarks")
    parser.add_argument(
        "wav",
        type=Path,
        nargs="?",
        help="Input wav for Whisper benchmark (omit when --translator)",
    )
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--compute-type",
        default="float16",
        choices=["float16", "int8", "int8_float16", "float32"],
    )
    parser.add_argument(
        "--translator",
        action="store_true",
        help="Run Qwen translator benchmark on canned business sentences",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"),
        help="Path to Qwen GGUF (with --translator)",
    )
    args = parser.parse_args()

    if args.translator:
        benchmark_translator(args.model_path, BUSINESS_SENTENCES)
        return 0

    if args.wav is None:
        parser.error("wav is required unless --translator is set")
    if not args.wav.exists():
        print(f"File not found: {args.wav}", file=sys.stderr)
        return 2

    benchmark_whisper(args.wav, args.model, args.device, args.compute_type)
    return 0


if __name__ == "__main__":
    sys.exit(main())
