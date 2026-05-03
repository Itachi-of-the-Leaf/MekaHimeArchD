# Amika's Ears (Architecture D)

High-performance, live audio diarization and separation engine.
Zero-latency capture via C++ bridge, ONNX-optimized inference, commercially safe model stack.

## Core Features

- **Zero-Latency Capture**: `miniaudio` + `nanobind` C++ bridge for 48kHz audio
- **Dual Speaker Separation**: SepFormer ONNX model (2-speaker output)
- **Dynamic Identity**: ECAPA-TDNN with EMA for continuous speaker learning
- **Local ASR**: `whisper.cpp` for fast, private transcription
- **High Concurrency**: Granian (Rust-based) ASGI web server
- **Latency Target**: <130ms end-to-end inference

---

## Model Sources & Licenses

| Model | Source | Code License | Training Data | Commercial Safe? |
|-------|--------|-------------|---------------|-----------------|
| **SepFormer** (source separation) | [speechbrain/sepformer-wsj02mix](https://huggingface.co/speechbrain/sepformer-wsj02mix) | Apache 2.0 | WSJ0-2mix (LDC) | ✅ Weights are Apache 2.0 |
| **ECAPA-TDNN** (speaker ID) | [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | Apache 2.0 | VoxCeleb1+2 | ✅ Fully safe |
| **Whisper** (ASR) | [whisper.cpp](https://github.com/ggerganov/whisper.cpp) | MIT | OpenAI Whisper (MIT) | ✅ Fully safe |

> **⚠️ WSJ0-2mix Data License Note**: The SepFormer model **weights** are released under Apache 2.0.
> However, the model was trained on WSJ0-2mix, which is derived from the LDC's proprietary WSJ0 corpus.
> Using pre-trained weights for inference is generally considered commercially safe.
> Do **NOT** retrain on or redistribute the WSJ0 training data without an LDC license.
> For a fully clean alternative, consider `sepformer-libri2mix` (trained on LibriSpeech, public domain).

---

## Project Structure

```
├── main.py                  # Entry point — Granian ASGI server
├── CMakeLists.txt           # Root CMake (delegates to src/bridge/)
├── pyproject.toml           # Python package config (scikit-build-core)
├── scripts/
│   ├── export_onnx.py       # Export PyTorch models → ONNX
│   └── test_pipeline.py     # Latency benchmark
├── src/
│   ├── bridge/
│   │   ├── audio_bridge.cpp # C++ capture bridge (miniaudio + nanobind)
│   │   ├── audio_bridge.py  # Python fallback (sounddevice)
│   │   └── CMakeLists.txt   # Bridge-level CMake (platform-aware)
│   ├── core/
│   │   ├── engine.py        # Main processing loop (separation + identity)
│   │   ├── models.py        # ONNX model wrappers (BSRNNSeparator, ECAPATDNNManager)
│   │   └── asr.py           # Whisper ASR interface
│   ├── database/
│   │   ├── manager.py       # SQLite identity storage
│   │   └── schema.sql       # Database schema
│   └── server/
│       └── app.py           # FastAPI app with WebSocket RMS streaming
├── models/                  # ONNX model files (gitignored)
│   ├── bsrnn.onnx           # SepFormer separator
│   ├── ecapa.onnx           # ECAPA-TDNN embeddings
│   └── ggml-base.en.bin     # Whisper model (whisper.cpp format)
├── include/
│   └── miniaudio.h          # Single-header audio library (gitignored, download separately)
└── pretrained_models/       # SpeechBrain checkpoints (auto-downloaded for export)
```

---

## Requirements

### Python
- **Python ≥ 3.10** (tested on 3.12, 3.14)
- All core dependencies support Python 3.14

### C++ Bridge (optional)
- **CMake ≥ 3.15**
- **C++ compiler**: MSVC (Windows), GCC/Clang (Linux/macOS)
- **nanobind** (installed via pip)

### System
- **No NVIDIA proprietary software required** — runs on CPU by default
- For GPU acceleration: install `onnxruntime-gpu` manually

---

## Setup

### 1. Install Python Dependencies

```bash
# Using uv (recommended):
uv sync

# Or with pip:
pip install .

# For model export (includes PyTorch, SpeechBrain):
pip install ".[export]"

# For GPU inference:
pip install ".[gpu]"
```

### 2. Export ONNX Models

```bash
# Export all models:
python scripts/export_onnx.py

# Export individually:
python scripts/export_onnx.py --ecapa
python scripts/export_onnx.py --sep
```

### 3. Build the C++ Bridge (optional)

The project includes a pure-Python fallback (`sounddevice`), so the C++ bridge
is optional. Build it for lower-latency capture:

#### Download miniaudio

```bash
# Download the single-header library:
mkdir include
curl -o include/miniaudio.h https://raw.githubusercontent.com/mackron/miniaudio/master/miniaudio.h
```

#### Windows (MSVC)

```powershell
cmake -B build -S . -G "Visual Studio 17 2022"
cmake --build build --config Release
```

#### Linux (GCC)

```bash
cmake -B build -S .
cmake --build build
```

#### macOS (Clang)

```bash
cmake -B build -S .
cmake --build build
```

The compiled module (`audio_bridge.pyd` on Windows, `audio_bridge.so` on Linux)
will be placed in `src/bridge/`.

### 4. Run the Server

```bash
python main.py
```

The server starts at `http://0.0.0.0:8000` with WebSocket RMS streaming at `ws://0.0.0.0:8000/`.

### 5. Test Pipeline Latency

```bash
python scripts/test_pipeline.py
```

Target: <130ms total inference per chunk.

---

## Architecture

```
Microphone → [C++ Bridge / sounddevice] → 48kHz f32 buffer
    ↓
[Accumulate 500ms chunks (24000 samples)]
    ↓
[Decimate 48kHz → 16kHz]
    ↓
[SepFormer ONNX] → Speaker 1 + Speaker 2 (16kHz)
    ↓
[OLA Crossfade] → Smooth speaker streams
    ↓
[ECAPA-TDNN ONNX] → Speaker embeddings → Identity matching (EMA)
    ↓
[SQLite DB] → Persistent identity storage
    ↓
[WebSocket] → Real-time RMS updates to frontend
```

---

## License

**MIT** — This project's code is MIT licensed.
Model licenses are listed in the table above.
