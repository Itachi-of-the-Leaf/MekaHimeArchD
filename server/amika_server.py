"""
server/amika_server.py — Phase 2: Decimation & VAD Gate

Pipeline per 20 ms frame (48 kHz mono L16 PCM):
  [WebSocket RX]
       │
       ├──► send_bytes(original 48kHz chunk) ──► [Client Echo — ALWAYS, FIRST]
       │
       └──► asyncio.create_task(  ← fire-and-forget, never blocks the RX loop
                asyncio.to_thread(
                    _vad_worker(chunk)   ← runs on thread-pool
                )
            )

_vad_worker:
  L16 bytes → float32 tensor
      → torchaudio.functional.resample(48kHz → 16kHz)
      → pad to 512 samples (Silero VAD v4 minimum chunk)
      → silero_vad_model(chunk, 16000)
      → print [SPEECH DETECTED] | [SILENCE]

Startup (Colab):
    python -m uvicorn server.amika_server:app --host 0.0.0.0 --port 8765

    (or from repo root)
    uvicorn server.amika_server:app --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("amika.server")

# ── Audio constants ───────────────────────────────────────────────────────────

SR_IN  = 48_000   # client capture rate
SR_VAD = 16_000   # Silero VAD rate
CHUNK_BYTES      = 1_920   # 20 ms @ 48kHz mono int16 = 960 samples × 2 bytes
SILERO_MIN_CHUNK = 512     # Silero VAD v4 minimum window at 16kHz
VAD_THRESHOLD    = 0.5     # speech confidence threshold

# ── Global model state (populated in lifespan) ────────────────────────────────

_vad_model: torch.nn.Module | None = None
_device: torch.device | None = None

# ── Silero VAD worker (runs on thread-pool, never touches the event loop) ─────

def _vad_worker(raw_bytes: bytes, frame_id: int) -> None:
    """
    Converts a 48kHz L16 PCM frame to 16kHz, runs Silero VAD inference,
    and logs the result. Intended to run via asyncio.to_thread().
    """
    assert _vad_model is not None, "VAD model not loaded"

    # 1. Parse L16 PCM → float32 (normalised to [-1, 1])
    pcm_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
    pcm_f32   = pcm_int16.astype(np.float32) / 32_768.0   # zero-copy view then cast

    # 2. Decimate 48kHz → 16kHz via torchaudio (no scipy dependency)
    tensor_48k = torch.from_numpy(pcm_f32).unsqueeze(0).to(_device)   # [1, 960]
    tensor_16k = torchaudio.functional.resample(
        tensor_48k, orig_freq=SR_IN, new_freq=SR_VAD
    )                                                                   # [1, 320]
    chunk = tensor_16k.squeeze(0)                                       # [320]

    # 3. Pad to Silero's minimum chunk size
    n = chunk.shape[0]
    if n < SILERO_MIN_CHUNK:
        chunk = torch.nn.functional.pad(chunk, (0, SILERO_MIN_CHUNK - n))

    # 4. VAD forward pass (no_grad — inference only)
    with torch.no_grad():
        confidence: float = _vad_model(chunk, SR_VAD).item()

    # 5. Human-readable telemetry
    label = "[SPEECH DETECTED]" if confidence > VAD_THRESHOLD else "[SILENCE]      "
    log.info("%s  conf=%.3f  frame=%d", label, confidence, frame_id)


# ── FastAPI lifespan: load model on startup ───────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _vad_model, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", _device)

    log.info("Loading Silero VAD v4 via torch.hub ...")
    # torch.hub.load is synchronous I/O — run on thread-pool so uvicorn startup
    # doesn't stall the event loop.
    _vad_model, _ = await asyncio.to_thread(
        torch.hub.load,
        "snakers4/silero-vad",
        "silero_vad",
        force_reload=False,
        onnx=False,
    )
    _vad_model = _vad_model.to(_device)
    _vad_model.eval()
    log.info("✓ Silero VAD v4 ready on %s.", _device)

    yield   # ── server is running ─────────────────────────────────────────────

    log.info("Server shutting down.")


app = FastAPI(title="Amika Server — Phase 2: VAD Gate", lifespan=lifespan)


# ── TCP_NODELAY helper (same as Phase 1) ─────────────────────────────────────

def _set_tcp_nodelay(websocket: WebSocket) -> None:
    try:
        transport: Any = websocket.scope.get("transport")
        if transport is None:
            transport = getattr(
                websocket._receive.__self__,  # type: ignore[attr-defined]
                "_transport",
                None,
            )
        if transport is not None:
            sock = transport.get_extra_info("socket")
            if sock is not None:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception as exc:  # noqa: BLE001
        log.debug("TCP_NODELAY: %s", exc)


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/audio")
async def audio_vad_gate(websocket: WebSocket) -> None:
    await websocket.accept()
    _set_tcp_nodelay(websocket)

    client = websocket.client
    log.info("Client connected: %s:%s", client.host, client.port)

    frame_count = 0
    t_start     = time.monotonic()

    try:
        while True:
            # ── 1. Receive 20ms binary frame (48kHz L16 PCM) ─────────────────
            data: bytes = await websocket.receive_bytes()
            frame_count += 1

            # ── 2. ECHO FIRST — original 48kHz chunk, unconditional ──────────
            await websocket.send_bytes(data)

            # ── 3. Fire VAD inference as a background task (non-blocking) ─────
            #     asyncio.create_task → schedules on event loop but doesn't wait
            #     asyncio.to_thread  → runs the CPU/GPU work on a thread-pool
            #     The receive loop immediately falls through to the next await.
            asyncio.create_task(
                asyncio.to_thread(_vad_worker, data, frame_count)
            )

            # ── 4. Periodic throughput telemetry (every 50 frames ≈ 1 s) ──────
            if frame_count % 50 == 0:
                elapsed = time.monotonic() - t_start
                log.info(
                    "[STATS] frames=%d  elapsed=%.1fs  avg_fps=%.1f",
                    frame_count, elapsed, frame_count / elapsed,
                )

    except WebSocketDisconnect:
        log.info("Client disconnected after %d frames.", frame_count)
    except Exception as exc:  # noqa: BLE001
        log.exception("Unexpected error: %s", exc)
    finally:
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()


# ── Health probe ──────────────────────────────────────────────────────────────

@app.get("/healthz")
async def healthz() -> dict:
    return {
        "status": "ok",
        "phase":  2,
        "device": str(_device),
        "vad_loaded": _vad_model is not None,
    }
