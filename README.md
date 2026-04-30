# Amika’s Ears (Architecture D)

High-performance, live audio diarization and separation engine.

## Core Features
- **Zero-Latency Capture**: `miniaudio` + `nanobind` C-bridge for 48kHz audio.
- **Dual Speaker Separation**: BSRNN/SepFormer optimized for ONNX.
- **Dynamic Identity**: ECAPA-TDNN with EMA for continuous speaker learning.
- **Local ASR**: `whisper.cpp` for fast, private transcription.
- **High Concurrency**: Granian (Rust-based) web server.

## Project Structure
- `src/bridge/`: C++ audio capture bridge.
- `src/core/`: Separation, Identity, and ASR logic.
- `src/server/`: Granian web server.
- `src/database/`: Identity and logging storage.
- `models/`: Place your `.onnx` and `.bin` models here.

## Setup
1. Install dependencies:
   ```bash
   pip install .
   ```
2. Build the C++ bridge:
   ```bash
   cd src/bridge
   mkdir build && cd build
   cmake ..
   make
   ```
3. Run the server:
   ```bash
   python -m src.server.app
   ```

## License
MIT
