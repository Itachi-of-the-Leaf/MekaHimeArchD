"""
server/amika_server.py — Phase 3: Decimation + VAD Gate + Speaker Identity

Per-frame pipeline (20 ms binary frame, 48 kHz mono L16 PCM):

  WebSocket RX
      │
      ├─► send_bytes(48kHz original)              ← ALWAYS FIRST, unconditional
      │
      └─► create_task(to_thread(_vad_worker))     ← fire-and-forget, non-blocking

  _vad_worker  [thread-pool]:
      1. int16 bytes → float32 → torchaudio.resample(48k→16k)
      2. Silero VAD v4 → confidence score
      3. If SPEECH:
           accumulate 16kHz PCM in per-connection ConnState buffer
           if buffer >= MIN_SPEECH_FRAMES or silence hangover triggered:
               wespeaker.extract_embedding → 256D vector
               SpeakerLibrary.identify_speaker → (name, score)
               log [SPEECH DETECTED] → Speaker: <NAME> (Conf: X.XX)
      4. If SILENCE:
           increment silence hangover counter
           log [SILENCE]

Startup (Colab / local):
    uvicorn server.amika_server:app --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import asyncio
import io
import logging
import socket
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from server.speaker_library import SpeakerLibrary

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("amika.server")

# ── Audio constants ───────────────────────────────────────────────────────────
SR_IN            = 48_000
SR_VAD           = 16_000
CHUNK_BYTES      = 1_920     # 20ms @ 48kHz mono int16
SILERO_MIN_CHUNK = 512       # Silero VAD v4 minimum samples at 16kHz
VAD_THRESHOLD    = 0.5

# ── Speech segment accumulator tunables ───────────────────────────────────────
MIN_SPEECH_FRAMES = 75       # 75 × 20ms = 1.5 s  (min for reliable embedding)
MAX_SPEECH_FRAMES = 150      # 150 × 20ms = 3.0 s  (force-extract if exceeded)
SILENCE_HANGOVER  = 5        # silent frames tolerated before segment closes

# ── Global model handles (set in lifespan) ────────────────────────────────────
_device:          torch.device | None        = None
_vad_model:       torch.nn.Module | None     = None
_speaker_model:   Any | None                 = None   # wespeaker Speaker object
_library:         SpeakerLibrary | None      = None

# ── Per-connection speech accumulation state ──────────────────────────────────
@dataclass
class ConnState:
    speech_16k:      list[np.ndarray] = field(default_factory=list)
    in_speech:       bool             = False
    silence_counter: int              = 0
    lock:            threading.Lock   = field(default_factory=threading.Lock)

_conn_states:      dict[str, ConnState] = {}
_conn_states_lock: threading.Lock       = threading.Lock()

# ── Model loader helpers (called in thread-pool during lifespan) ──────────────

def _load_vad() -> torch.nn.Module:
    model, _ = torch.hub.load(
        "snakers4/silero-vad", "silero_vad",
        force_reload=False, onnx=False,
    )
    return model

def _load_wespeaker() -> Any:
    import wespeaker as _ws
    model = _ws.load_model("english")       # ResNet34, 256D, trained on VoxCeleb2
    if _device is not None and _device.type == "cuda":
        model.set_gpu(0)
    else:
        model.set_cpu()
    return model

# ── VAD worker (runs on thread-pool, never touches the event loop) ────────────

def _extract_embedding(speech_f32: np.ndarray) -> np.ndarray:
    """Write accumulated 16kHz PCM to in-memory WAV and call wespeaker."""
    tensor = torch.from_numpy(speech_f32).unsqueeze(0)   # [1, T]
    buf    = io.BytesIO()
    torchaudio.save(buf, tensor, SR_VAD, format="WAV")
    buf.seek(0)
    emb = _speaker_model.extract_embedding(buf)           # returns np.ndarray or Tensor
    if isinstance(emb, torch.Tensor):
        emb = emb.squeeze().cpu().numpy()
    return np.array(emb, dtype=np.float32).flatten()


def _vad_worker(raw_bytes: bytes, frame_id: int, conn_key: str) -> None:
    """Core per-frame processing: decimation → VAD → optional speaker ID."""

    # ── 1. Decode + decimate 48kHz → 16kHz ───────────────────────────────────
    pcm_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
    pcm_f32   = pcm_int16.astype(np.float32) / 32_768.0

    t48 = torch.from_numpy(pcm_f32).unsqueeze(0).to(_device)   # [1, 960]
    t16 = torchaudio.functional.resample(t48, SR_IN, SR_VAD)   # [1, 320]
    chunk_16 = t16.squeeze(0)                                    # [320]

    # ── 2. Silero VAD ─────────────────────────────────────────────────────────
    padded = chunk_16
    if chunk_16.shape[0] < SILERO_MIN_CHUNK:
        padded = torch.nn.functional.pad(
            chunk_16, (0, SILERO_MIN_CHUNK - chunk_16.shape[0])
        )

    with torch.no_grad():
        vad_conf: float = _vad_model(padded, SR_VAD).item()

    is_speech = vad_conf > VAD_THRESHOLD

    # ── 3. Fetch per-connection accumulator ───────────────────────────────────
    with _conn_states_lock:
        state = _conn_states.get(conn_key)
    if state is None:
        # Connection already closed — silently discard
        return

    # ── 4. Accumulate speech / detect segment boundaries ─────────────────────
    speech_to_embed: np.ndarray | None = None

    with state.lock:
        if is_speech:
            state.in_speech      = True
            state.silence_counter = 0
            state.speech_16k.append(chunk_16.cpu().numpy())

            # Force-extract if we've accumulated enough for a reliable embedding
            if len(state.speech_16k) >= MAX_SPEECH_FRAMES:
                speech_to_embed   = np.concatenate(state.speech_16k)
                state.speech_16k  = []
        else:
            if state.in_speech:
                state.silence_counter += 1
                # Close the segment once hangover exhausted AND we have enough speech
                if (state.silence_counter >= SILENCE_HANGOVER
                        and len(state.speech_16k) >= MIN_SPEECH_FRAMES):
                    speech_to_embed   = np.concatenate(state.speech_16k)
                    state.speech_16k  = []
                    state.in_speech   = False
                    state.silence_counter = 0

    # ── 5. Speaker identification (only when a full segment is ready) ─────────
    if speech_to_embed is not None and _speaker_model is not None:
        try:
            emb    = _extract_embedding(speech_to_embed)
            result = _library.identify_speaker(emb)
            log.info(
                "[SPEECH DETECTED] → Speaker: %s (Conf: %.2f) | frame=%d",
                result.name, result.score, frame_id,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("[SPEECH] Embedding extraction failed: %s", exc)
    elif is_speech:
        log.info("[SPEECH DETECTED]  conf=%.3f | frame=%d", vad_conf, frame_id)
    else:
        log.info("[SILENCE]          conf=%.3f | frame=%d", vad_conf, frame_id)


# ── FastAPI lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _device, _vad_model, _speaker_model, _library

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", _device)

    log.info("Loading Silero VAD v4 ...")
    _vad_model = await asyncio.to_thread(_load_vad)
    _vad_model = _vad_model.to(_device).eval()
    log.info("✓ Silero VAD v4 ready.")

    log.info("Loading Wespeaker ResNet34 ...")
    _speaker_model = await asyncio.to_thread(_load_wespeaker)
    log.info("✓ Wespeaker ResNet34 ready on %s.", _device)

    _library = SpeakerLibrary(data_dir=Path(__file__).parent)
    log.info("✓ SpeakerLibrary ready (%d speakers).", len(_library._names))

    yield

    log.info("Server shutting down.")


app = FastAPI(title="Amika Server — Phase 3: VAD + Speaker Identity",
              lifespan=lifespan)


# ── TCP_NODELAY helper ────────────────────────────────────────────────────────

def _set_tcp_nodelay(websocket: WebSocket) -> None:
    try:
        transport: Any = websocket.scope.get("transport")
        if transport is None:
            transport = getattr(
                websocket._receive.__self__,  # type: ignore[attr-defined]
                "_transport", None,
            )
        if transport is not None:
            sock = transport.get_extra_info("socket")
            if sock:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception as exc:  # noqa: BLE001
        log.debug("TCP_NODELAY: %s", exc)


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/audio")
async def audio_pipeline(websocket: WebSocket) -> None:
    await websocket.accept()
    _set_tcp_nodelay(websocket)

    client    = websocket.client
    conn_key  = f"{client.host}:{client.port}"
    log.info("Client connected: %s", conn_key)

    # Register per-connection state
    with _conn_states_lock:
        _conn_states[conn_key] = ConnState()

    frame_count = 0
    t_start     = time.monotonic()

    try:
        while True:
            data: bytes = await websocket.receive_bytes()
            frame_count += 1

            # ── ECHO FIRST: original 48kHz chunk, always, unconditional ───────
            await websocket.send_bytes(data)

            # ── Fire VAD+ID as background task (non-blocking) ─────────────────
            asyncio.create_task(
                asyncio.to_thread(_vad_worker, data, frame_count, conn_key)
            )

            if frame_count % 50 == 0:
                elapsed = time.monotonic() - t_start
                log.info("[STATS] frames=%d  elapsed=%.1fs  fps=%.1f",
                         frame_count, elapsed, frame_count / elapsed)

    except WebSocketDisconnect:
        log.info("Client disconnected: %s (%d frames)", conn_key, frame_count)
    except Exception as exc:  # noqa: BLE001
        log.exception("Unexpected error on %s: %s", conn_key, exc)
    finally:
        with _conn_states_lock:
            _conn_states.pop(conn_key, None)
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()


# ── Health probe ──────────────────────────────────────────────────────────────

@app.get("/healthz")
async def healthz() -> dict:
    return {
        "status":          "ok",
        "phase":           3,
        "device":          str(_device),
        "vad_loaded":      _vad_model is not None,
        "wespeaker_loaded": _speaker_model is not None,
        "enrolled_speakers": len(_library._names) if _library else 0,
    }
