"""Phase 2 open-question probe: discover VibeVoice-ASR-7B's API surface.

Goal: load microsoft/VibeVoice-ASR-7B with the lightest possible
download, inspect what classes / methods / kwargs it exposes, and
attempt a 30-second transcription on a wav file. Prints everything we
need to rewrite docs/superpowers/specs/2026-04-30-meet-mirror-phase2-notes-design.md
§6.1 with the real API instead of our guess.

Does NOT modify the project. Purely informational. Re-runnable.

Usage:

    pip install transformers accelerate bitsandbytes
    python scripts/probe_vibevoice.py path/to/30sec_english.wav

If you don't have a 30-s clip handy, point it at any session audio:

    python scripts/probe_vibevoice.py sessions/2026-04-30_13-20-51/audio.wav
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO = "microsoft/VibeVoice-ASR-7B"


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def probe_repo_listing() -> None:
    section("1. HF repo file listing")
    try:
        from huggingface_hub import HfApi

        files = HfApi().list_repo_files(REPO)
        print(f"Repo: {REPO}")
        print(f"Files ({len(files)}):")
        for f in files:
            print(f"  {f}")
    except Exception as e:
        print(f"  ERROR listing repo: {e}")
        print("  (If 404, the repo name in the spec is wrong — fix that first.)")


def probe_config() -> None:
    section("2. Model config")
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
        print(f"  type:           {type(cfg).__name__}")
        print(f"  model_type:     {getattr(cfg, 'model_type', '?')}")
        print(f"  architectures:  {getattr(cfg, 'architectures', '?')}")
        # Dump shortlist of interesting fields
        for k in (
            "max_position_embeddings",
            "audio_max_length",
            "num_speakers",
            "vocab_size",
            "hidden_size",
        ):
            if hasattr(cfg, k):
                print(f"  {k}: {getattr(cfg, k)}")
    except Exception as e:
        print(f"  ERROR loading config: {e}")


def probe_processor() -> None:
    section("3. Processor / tokenizer")
    try:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(REPO, trust_remote_code=True)
        print(f"  type: {type(proc).__name__}")
        print(f"  attrs: {[a for a in dir(proc) if not a.startswith('_')][:30]}")
        if hasattr(proc, "feature_extractor"):
            print(f"  feature_extractor: {type(proc.feature_extractor).__name__}")
        if hasattr(proc, "tokenizer"):
            print(f"  tokenizer: {type(proc.tokenizer).__name__}")
    except Exception as e:
        print(f"  ERROR loading processor: {e}")
        print("    (model may use AutoFeatureExtractor / AutoTokenizer separately)")


def probe_model_load(bits: int) -> object | None:
    section(f"4. Model load (4-bit={bits == 4}, 8-bit={bits == 8})")
    try:
        import torch
        from transformers import AutoModel

        kwargs: dict = {"trust_remote_code": True}
        if bits == 4:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            kwargs["device_map"] = "auto"
        elif bits == 8:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"

        t0 = time.perf_counter()
        model = AutoModel.from_pretrained(REPO, **kwargs)
        print(f"  Load: {(time.perf_counter() - t0):.1f}s")
        print(f"  Class: {type(model).__name__}")
        print(f"  Device(s): {set(p.device for p in model.parameters())}")

        gen_methods = [m for m in dir(model) if "generat" in m.lower()]
        print(f"  generate-like methods: {gen_methods}")

        for cls in type(model).__mro__:
            try:
                import inspect

                src = inspect.getsourcefile(cls)
            except Exception:
                src = "?"
            print(f"  MRO: {cls.__module__}.{cls.__name__}  ({src})")

        if hasattr(model, "generate"):
            import inspect

            try:
                sig = inspect.signature(model.generate)
                print(f"  generate signature: {sig}")
            except (TypeError, ValueError):
                print("  generate signature: <not introspectable>")

        if hasattr(model, "config"):
            print("  Config kwargs that might matter:")
            for k in (
                "task",
                "task_token_id",
                "speaker_token_id",
                "diarize",
                "include_timestamps",
            ):
                if hasattr(model.config, k):
                    print(f"    config.{k} = {getattr(model.config, k)}")

        return model
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        traceback.print_exc()
        return None


def probe_inference(model: object, wav_path: Path) -> None:
    section("5. Single-clip inference attempts")

    try:
        import numpy as np
        import soundfile as sf
        import torch
    except Exception as e:
        print(f"  ERROR loading audio deps: {e}")
        return

    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        print(f"  WARNING sr={sr} != 16000; resample externally if needed")
    print(f"  Loaded {wav_path.name}: {len(audio) / sr:.1f}s @ {sr}Hz")

    # Trim to first 30 s to stay within VibeVoice's window
    audio = audio[: 30 * sr]

    try:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(REPO, trust_remote_code=True)
    except Exception as e:
        print(f"  Could not load processor: {e}")
        return

    # Try a few common encoding shapes
    inputs = None
    for attempt in ("processor(audio, sampling_rate=sr)", "processor(audio=audio, sampling_rate=sr)"):
        try:
            print(f"  trying: {attempt}")
            if "audio=" in attempt:
                inputs = proc(audio=audio, sampling_rate=sr, return_tensors="pt")
            else:
                inputs = proc(audio, sampling_rate=sr, return_tensors="pt")
            print(f"    OK: keys={list(inputs.keys())}")
            for k, v in inputs.items():
                shape = getattr(v, "shape", None)
                print(f"    {k}.shape = {shape}")
            break
        except Exception as e:
            print(f"    FAIL: {e}")

    if inputs is None:
        print("  could not encode audio — stop here, fix processor call shape first")
        return

    # Move to GPU
    inputs = {k: v.to("cuda") for k, v in inputs.items() if hasattr(v, "to")}

    # Try generate with multiple kwargs
    candidate_kwargs: list[dict] = [
        {"max_new_tokens": 1024},
        {"max_new_tokens": 1024, "task": "transcribe"},
        {"max_new_tokens": 1024, "include_speaker_labels": True},
        {"max_new_tokens": 1024, "return_diarization": True},
        {"max_new_tokens": 1024, "task": "transcribe", "return_timestamps": True},
        {"max_new_tokens": 1024, "task": "transcribe", "return_timestamps": True,
         "return_speakers": True},
    ]

    for kw in candidate_kwargs:
        try:
            print(f"\n  generate(**inputs, **{kw})")
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**inputs, **kw)  # type: ignore[attr-defined]
            print(f"    OK in {time.perf_counter() - t0:.1f}s")
            print(f"    output type: {type(out).__name__}")
            if hasattr(out, "shape"):
                print(f"    output shape: {out.shape}")
                # Try to decode
                try:
                    text = proc.batch_decode(out, skip_special_tokens=True)
                    print(f"    decoded: {text[0][:200]}{'...' if len(text[0])>200 else ''}")
                except Exception as e:
                    print(f"    decode failed: {e}")
            else:
                print(f"    output repr (first 500 chars): {repr(out)[:500]}")
            # Stop on first success — that's the working API
            return
        except TypeError as e:
            # kwarg not accepted; keep trying
            print(f"    kwarg rejected: {e}")
        except Exception as e:
            print(f"    runtime error: {e}")
            traceback.print_exc()
            return


def probe_vram() -> None:
    section("6. VRAM snapshot")
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"  free:  {free / 1e9:.2f} GB")
            print(f"  total: {total / 1e9:.2f} GB")
            print(f"  used:  {(total - free) / 1e9:.2f} GB")
        else:
            print("  CUDA not available")
    except Exception as e:
        print(f"  ERROR: {e}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=Path, nargs="?", help="30 s English wav for inference test")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8, 16])
    parser.add_argument("--skip-load", action="store_true", help="Only probe repo + config + processor")
    args = parser.parse_args()

    print(f"Probing {REPO}\n")

    probe_repo_listing()
    probe_config()
    probe_processor()

    if args.skip_load:
        print("\n--skip-load: not loading model")
        return 0

    model = probe_model_load(args.bits)

    if model is not None and args.wav is not None and args.wav.exists():
        probe_inference(model, args.wav)

    probe_vram()

    section("Summary")
    print(json.dumps({
        "repo": REPO,
        "bits": args.bits,
        "wav": str(args.wav) if args.wav else None,
        "model_loaded": model is not None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
