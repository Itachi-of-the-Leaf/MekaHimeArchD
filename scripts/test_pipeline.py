"""
End-to-End Pipeline Latency Test for Amika's Ears (Architecture D)

Tests ONNX model loading and inference with synthetic audio.
Target: <130ms total inference latency.

Usage:
    python scripts/test_pipeline.py
"""

import os
import sys
import time

import numpy as np


def _check_model(path: str, name: str) -> bool:
    """Check if a model file exists (including external data format)."""
    if os.path.isfile(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  [FOUND] {name}: {path} ({size_mb:.1f} MB)")
        return True
    data_path = path + ".data"
    if os.path.isfile(data_path):
        size_mb = os.path.getsize(data_path) / (1024 * 1024)
        print(f"  [FOUND] {name}: {path} + .data ({size_mb:.1f} MB)")
        return True
    print(f"  [MISSING] {name}: {path}")
    return False


def test_separator(model_path: str = "models/bsrnn.onnx") -> dict[str, float]:
    """Test source separation model loading and inference."""
    import onnxruntime as ort

    print("\n" + "=" * 60)
    print("Source Separator (SepFormer) -- Latency Test")
    print("=" * 60)

    results: dict[str, float] = {}

    # 1. Load model
    t0 = time.perf_counter()
    providers = ["CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        print(f"  [FAIL] Cannot load {model_path}: {e}")
        return {"sep_load_ms": -1, "sep_inference_ms": -1}

    t_load = (time.perf_counter() - t0) * 1000
    results["sep_load_ms"] = t_load
    print(f"  Model loaded in {t_load:.1f}ms")

    # Print input/output info
    for inp in session.get_inputs():
        print(f"  Input:  {inp.name} -- shape={inp.shape}, dtype={inp.type}")
    for out in session.get_outputs():
        print(f"  Output: {out.name} -- shape={out.shape}, dtype={out.type}")

    # 2. Generate synthetic audio (500ms @ 16kHz = 8000 samples)
    # This matches the pipeline's chunk size after 48->16kHz decimation
    audio_16k = np.random.randn(1, 8000).astype(np.float32) * 0.1

    input_name = session.get_inputs()[0].name

    # 3. Warmup run
    try:
        _ = session.run(None, {input_name: audio_16k})
    except Exception as e:
        print(f"  [FAIL] Inference failed: {e}")
        return results

    # 4. Benchmark (10 iterations)
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: audio_16k})
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)
    results["sep_inference_ms"] = avg_ms

    print(f"\n  Inference (10 runs):")
    print(f"    Avg: {avg_ms:.1f}ms | Min: {min_ms:.1f}ms | Max: {max_ms:.1f}ms")
    print(f"    Output shape: {outputs[0].shape}")

    return results


def test_ecapa(model_path: str = "models/ecapa.onnx") -> dict[str, float]:
    """Test speaker embedding model loading and inference."""
    import onnxruntime as ort

    print("\n" + "=" * 60)
    print("Speaker Embeddings (ECAPA-TDNN) -- Latency Test")
    print("=" * 60)

    results: dict[str, float] = {}

    # 1. Load model
    t0 = time.perf_counter()
    providers = ["CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        print(f"  [FAIL] Cannot load {model_path}: {e}")
        return {"ecapa_load_ms": -1, "ecapa_inference_ms": -1}

    t_load = (time.perf_counter() - t0) * 1000
    results["ecapa_load_ms"] = t_load
    print(f"  Model loaded in {t_load:.1f}ms")

    # Print input/output info
    for inp in session.get_inputs():
        print(f"  Input:  {inp.name} -- shape={inp.shape}, dtype={inp.type}")
    for out in session.get_outputs():
        print(f"  Output: {out.name} -- shape={out.shape}, dtype={out.type}")

    # 2. Generate synthetic features (100 frames x 80 mel channels)
    features = np.random.randn(1, 100, 80).astype(np.float32)
    input_name = session.get_inputs()[0].name

    # 3. Warmup
    try:
        _ = session.run(None, {input_name: features})
    except Exception as e:
        print(f"  [FAIL] Inference failed: {e}")
        return results

    # 4. Benchmark (10 iterations)
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: features})
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)
    results["ecapa_inference_ms"] = avg_ms

    print(f"\n  Inference (10 runs):")
    print(f"    Avg: {avg_ms:.1f}ms | Min: {min_ms:.1f}ms | Max: {max_ms:.1f}ms")
    print(f"    Output shape: {outputs[0].shape}")

    return results


def main() -> None:
    print("+" + "=" * 58 + "+")
    print("|    Amika's Ears -- Pipeline Latency Benchmark            |")
    print("|    Architecture D -- Target <130ms                       |")
    print("+" + "=" * 58 + "+")

    # Check models exist
    print("\nModel Status:")
    sep_ok = _check_model("models/bsrnn.onnx", "SepFormer")
    ecapa_ok = _check_model("models/ecapa.onnx", "ECAPA-TDNN")

    if not sep_ok and not ecapa_ok:
        print("\n[ABORT] No ONNX models found. Run export first:")
        print("  python scripts/export_onnx.py")
        sys.exit(1)

    all_results: dict[str, float] = {}

    # Test available models
    if sep_ok:
        all_results.update(test_separator())
    if ecapa_ok:
        all_results.update(test_ecapa())

    # Summary Table
    print("\n" + "=" * 60)
    print("LATENCY SUMMARY")
    print("=" * 60)
    print(f"  {'Metric':<28s} {'Time (ms)':>10s} {'Status':>8s}")
    print("  " + "-" * 50)

    total_inference = 0.0
    target_ms = 130.0

    for key, val in sorted(all_results.items()):
        if val < 0:
            status = "--"
        elif "load" in key:
            status = "--"
        else:
            total_inference += val
            status = "OK" if val < target_ms else "SLOW"
        print(f"  {key:<28s} {val:>10.1f}ms {status:>8s}")

    print("-" * 54)
    combined_status = "PASS" if total_inference < target_ms else "FAIL"
    print(f"  {'TOTAL INFERENCE':<28s} {total_inference:>10.1f}ms {combined_status:>8s}")
    print(f"  {'Target':<28s} {target_ms:>10.1f}ms")
    print()

    if total_inference >= target_ms:
        print(f"[WARNING] Total inference ({total_inference:.1f}ms) exceeds target ({target_ms}ms).")
        print("  Consider: GPU acceleration (onnxruntime-gpu), model quantization, or smaller chunk sizes.")
        sys.exit(1)
    else:
        print(f"[PASS] Total inference ({total_inference:.1f}ms) is within the {target_ms}ms target.")


if __name__ == "__main__":
    main()
