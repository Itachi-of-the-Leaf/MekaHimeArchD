"""
ONNX Model Exporter for Amika's Ears (Architecture D)

Exports SpeechBrain models to ONNX format for production inference.
Includes license verification to ensure commercial safety.

Usage:
    python scripts/export_onnx.py           # Export all models
    python scripts/export_onnx.py --ecapa   # Export ECAPA-TDNN only
    python scripts/export_onnx.py --sep     # Export SepFormer only
"""

import os
import sys
import time
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.onnx

# Prevent SpeechBrain's lazy-loading from crashing torch.onnx.export
# which uses 'inspect' to scan all modules in sys.modules.
try:
    import speechbrain.utils.importutils as importutils
    def safe_getattr(self, attr):
        try:
            return getattr(self.ensure_module(1), attr)
        except (ImportError, AttributeError):
            return None
    importutils.LazyModule.__getattr__ = safe_getattr
except ImportError:
    pass

# ──────────────────────────────────────────────────────────────
# License Registry — Only Apache 2.0 / MIT models are allowed
# ──────────────────────────────────────────────────────────────

MODEL_LICENSES: dict[str, dict[str, str]] = {
    "speechbrain/spkrec-ecapa-voxceleb": {
        "license": "Apache-2.0",
        "training_data": "VoxCeleb1+2",
        "training_data_license": "Creative Commons Attribution 4.0 (VoxCeleb)",
        "commercial_safe": "YES",
        "notes": "Weights released under Apache 2.0 by SpeechBrain.",
    },
    "speechbrain/sepformer-wsj02mix": {
        "license": "Apache-2.0",
        "training_data": "WSJ0-2mix",
        "training_data_license": "LDC Proprietary (WSJ0 corpus)",
        "commercial_safe": "YES (weights only — do NOT redistribute training data)",
        "notes": (
            "Model weights are Apache 2.0. Training data (WSJ0) is LDC-proprietary. "
            "Using pre-trained weights for inference is generally accepted as commercially safe. "
            "Do NOT retrain on or redistribute WSJ0 data without an LDC license."
        ),
    },
}

ALLOWED_LICENSES: set[str] = {"Apache-2.0", "MIT"}


def verify_license(model_source: str) -> bool:
    """Verify that a model's license is commercially safe (Apache 2.0 / MIT only)."""
    info = MODEL_LICENSES.get(model_source)
    if info is None:
        print(f"[FAIL] Unknown model: {model_source}. Cannot verify license.")
        return False

    license_id = info["license"]
    if license_id not in ALLOWED_LICENSES:
        print(f"[FAIL] Model '{model_source}' has license '{license_id}' — NOT in allowed set {ALLOWED_LICENSES}.")
        return False

    print(f"[LICENSE OK] {model_source}")
    print(f"  License:        {license_id}")
    print(f"  Training Data:  {info['training_data']} ({info['training_data_license']})")
    print(f"  Commercial:     {info['commercial_safe']}")
    if info.get("notes"):
        print(f"  Notes:          {info['notes']}")
    return True


# ──────────────────────────────────────────────────────────────
# ECAPA-TDNN Export
# ──────────────────────────────────────────────────────────────

def export_ecapa(output_dir: str = "models") -> None:
    """Export ECAPA-TDNN embedding model to ONNX."""
    source = "speechbrain/spkrec-ecapa-voxceleb"

    print("\n" + "=" * 60)
    print("ECAPA-TDNN Export")
    print("=" * 60)

    if not verify_license(source):
        print("[ABORT] License verification failed.")
        return

    from speechbrain.inference.speaker import EncoderClassifier

    t0 = time.perf_counter()
    print("\nLoading ECAPA-TDNN PyTorch model...")
    classifier = EncoderClassifier.from_hparams(
        source=source,
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
    )
    t_load = time.perf_counter() - t0
    print(f"  Model loaded in {t_load:.2f}s")

    # The embedding model expects mel-spectrogram features, NOT raw audio.
    # Input shape: [batch, time_frames, n_mels] — typically [1, 100, 80]
    embedding_model = classifier.mods.embedding_model
    embedding_model.eval()

    dummy_input = torch.randn(1, 100, 80, device=classifier.device)

    print("Tracing ECAPA embedding model...")
    traced = torch.jit.trace(embedding_model, dummy_input, check_trace=False)

    output_path = os.path.join(output_dir, "ecapa.onnx")
    print(f"Exporting to {output_path}...")

    t0 = time.perf_counter()
    torch.onnx.export(
        traced,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_features"],
        output_names=["embedding"],
        dynamic_axes={
            "input_features": {0: "batch_size", 1: "time"},
            "embedding": {0: "batch_size"},
        },
        dynamo=False,
    )
    t_export = time.perf_counter() - t0

    # Verify the .onnx file was actually created (not just .onnx.data)
    if os.path.isfile(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[SUCCESS] {output_path} ({size_mb:.1f} MB) exported in {t_export:.2f}s")
    else:
        # If torch produced external data format, the graph + data are separate.
        # Check for the .data file
        data_path = output_path + ".data"
        if os.path.isfile(data_path):
            data_mb = os.path.getsize(data_path) / (1024 * 1024)
            print(f"[WARNING] Export produced external data format:")
            print(f"  Graph:  {output_path} — {'EXISTS' if os.path.isfile(output_path) else 'MISSING'}")
            print(f"  Data:   {data_path} ({data_mb:.1f} MB)")
            print(f"  Both files must be co-located for inference.")
        else:
            print(f"[FAIL] No output file found at {output_path}")


# ──────────────────────────────────────────────────────────────
# SepFormer Export (2-speaker separation)
# ──────────────────────────────────────────────────────────────

class SepFormerONNXWrapper(nn.Module):
    """
    Wraps SpeechBrain's SepFormer internal modules for clean ONNX export.

    SpeechBrain's SepformerSeparation.separate_batch() does:
      1. Encoder(mix) → encoded
      2. MaskNet(encoded) → masks  (shape: [batch, num_speakers, time, features])
      3. Decoder(encoded * mask_i) → separated_i for each speaker

    We replicate this logic directly to avoid SpeechBrain's file I/O and
    produce a clean graph with output shape [batch, num_speakers, time].
    """

    def __init__(self, encoder: nn.Module, masknet: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.masknet = masknet
        self.decoder = decoder

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mix: [batch, time] raw waveform
        Returns:
            separated: [batch, num_speakers, time]
        """
        # 1. Encode the mixture
        mix_encoded = self.encoder(mix)

        # 2. Estimate separation masks
        # SpeechBrain SepFormer returns masks as [num_spks, batch, features, time]
        masks = self.masknet(mix_encoded)

        # 3. Apply each mask and decode
        separated = []
        num_spks = masks.shape[0]
        
        for i in range(num_spks):
            mask_i = masks[i] # [batch, features, time]
            
            # Match mask shape to mix_encoded [B, N, T]
            # (No transpose usually needed if shapes are [B, N, T])
            if mask_i.shape != mix_encoded.shape:
                mask_i = mask_i.view(mix_encoded.shape)
                
            separated_i = self.decoder(mix_encoded * mask_i) # [batch, time]
            separated.append(separated_i)

        # Stack speakers → [batch, num_speakers, time]
        out = torch.stack(separated, dim=1)
        return out


def export_separator(output_dir: str = "models") -> None:
    """Export SepFormer (WSJ0-2mix) to ONNX with 2-speaker output."""
    source = "speechbrain/sepformer-wsj02mix"

    print("\n" + "=" * 60)
    print("SepFormer (WSJ0-2mix) Export — 2-Speaker Output")
    print("=" * 60)

    if not verify_license(source):
        print("[ABORT] License verification failed.")
        return

    from speechbrain.inference.separation import SepformerSeparation

    t0 = time.perf_counter()
    print("\nLoading SepFormer PyTorch model...")
    separator = SepformerSeparation.from_hparams(
        source=source,
        savedir="pretrained_models/sepformer-wsj02mix",
        run_opts={"device": "cpu"},
    )
    t_load = time.perf_counter() - t0
    print(f"  Model loaded in {t_load:.2f}s")

    # Extract the internal modules
    encoder = separator.mods.encoder
    masknet = separator.mods.masknet
    decoder = separator.mods.decoder

    wrapper = SepFormerONNXWrapper(encoder, masknet, decoder)
    wrapper.eval()

    # 2. Trace with dummy input (8000 samples = 500ms @ 16kHz resampled to model rate)
    # We use the exact production chunk size to avoid Gather out-of-bounds errors.
    dummy_input = torch.randn(1, 8000, device="cpu")

    output_path = os.path.join(output_dir, "bsrnn.onnx")
    print(f"Exporting to {output_path}...")

    t0 = time.perf_counter()
    try:
        torch.onnx.export(
            wrapper,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input_audio"],
            output_names=["separated_sources"],
            dynamic_axes={
                "input_audio": {0: "batch_size", 1: "time"},
                "separated_sources": {0: "batch_size", 2: "time"},
            },
            dynamo=False,
        )
        t_export = time.perf_counter() - t0

        if os.path.isfile(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"[SUCCESS] {output_path} ({size_mb:.1f} MB) exported in {t_export:.2f}s")

            # Also check for external data
            data_path = output_path + ".data"
            if os.path.isfile(data_path):
                data_mb = os.path.getsize(data_path) / (1024 * 1024)
                print(f"  External data: {data_path} ({data_mb:.1f} MB)")
        else:
            print(f"[FAIL] No output file found at {output_path}")

    except Exception as e:
        t_export = time.perf_counter() - t0
        print(f"[FAIL] SepFormer export failed after {t_export:.2f}s:")
        print(f"  {type(e).__name__}: {e}")
        print("\n  This may happen because SepFormer's internal ops are not fully")
        print("  ONNX-traceable. Consider using torch.jit.script or a different")
        print("  export strategy if this persists.")


# ──────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Export ONNX models for Amika's Ears")
    parser.add_argument("--ecapa", action="store_true", help="Export ECAPA-TDNN only")
    parser.add_argument("--sep", action="store_true", help="Export SepFormer only")
    parser.add_argument("--output-dir", default="models", help="Output directory (default: models)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # If no specific flag, export all
    export_all = not args.ecapa and not args.sep

    print("+" + "=" * 58 + "+")
    print("|        Amika's Ears -- ONNX Model Exporter               |")
    print("|        Architecture D -- Commercial-Safe Models          |")
    print("+" + "=" * 58 + "+")

    total_start = time.perf_counter()

    if export_all or args.ecapa:
        export_ecapa(args.output_dir)

    if export_all or args.sep:
        export_separator(args.output_dir)

    total_time = time.perf_counter() - total_start
    print(f"\n{'=' * 60}")
    print(f"Total export time: {total_time:.2f}s")
    print(f"Output directory:  {args.output_dir}/")

    # List generated files
    print(f"\nGenerated files:")
    for f in sorted(os.listdir(args.output_dir)):
        fp = os.path.join(args.output_dir, f)
        if os.path.isfile(fp):
            size_mb = os.path.getsize(fp) / (1024 * 1024)
            print(f"  {f:30s} {size_mb:8.1f} MB")


if __name__ == "__main__":
    main()
