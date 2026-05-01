"""
server/amika_server.py — Phase 3: VAD Gate + Speech Accumulator + Speaker Identity

Transport : Binary WebSocket, 20 ms L16 PCM frames @ 48 kHz mono.
ASGI server: granian  (NOT uvicorn)

Startup (Colab / local — from repo root):
    granian --interface asgi server.amika_server:app --host 0.0.0.0 --port 8765

Per-frame pipeline:
  WebSocket RX  (48kHz L16 binary frame)
      │
      ├─► send_bytes(original 48kHz chunk)          ← ALWAYS FIRST, unconditional
      │
      └─► create_task(to_thread(_vad_worker))        ← fire-and-forget, non-blocking

  _vad_worker [thread-pool]:
      1. int16 → float32 → torchaudio.resample(48k→16k)   [Decimation Node]
      2. Silero VAD v4 → confidence score                   [VAD Gate]
      3a. SPEECH  → append 16kHz chunk to ConnState buffer
                    if buffer >= 2.0s OR silence hangover:
                        wespeaker.extract_embedding(buffer)
                        SpeakerLibrary.identify_speaker(emb)
                        log [IDENTITY] → Speaker: <NAME> (Conf: X.XX)
                        clear buffer
      3b. SILENCE → log [SILENCE], increment hangover counter

Echo is unconditional and always precedes all inference — inference never
delays the raw audio return to the client.
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
CHUNK_BYTES      = 1_920      # 20 ms @ 48 kHz mono int16  = 960 samples × 2 bytes
SILERO_MIN_CHUNK = 512        # Silero VAD v4 minimum window at 16 kHz
VAD_THRESHOLD    = 0.5

# ── Speech accumulator tunables ───────────────────────────────────────────────
SPEECH_SEGMENT_FRAMES = 100   # 100 × 20 ms = 2.0 s  (trigger embedding extraction)
SILENCE_HANGOVER      = 5     # allow 5 silent frames (100 ms) before closing segment

# ── Global model handles (populated in lifespan) ──────────────────────────────
_device:        torch.device | None    = None
_vad_model:     torch.nn.Module | None = None
_speaker_model: Any | None             = None   # wespeaker.Speaker object
_library:       SpeakerLibrary | None  = None


# ── Per-connection state ───────────────────────────────────────────────────────
@dataclass
class ConnState:
    """Buffers 16 kHz speech frames until a full segment is ready."""
    speech_16k:      list[np.ndarray] = field(default_factory=list)
    in_speech:       bool             = False
    silence_counter: int              = 0
    lock:            threading.Lock   = field(default_factory=threading.Lock)


_conn_states:      dict[str, ConnState] = {}
_conn_states_lock: threading.Lock       = threading.Lock()


# ── Model loaders (run in thread-pool during lifespan) ────────────────────────

def _load_vad() -> torch.nn.Module:
    model, _ = torch.hub.load(
        "snakers4/silero-vad", "silero_vad",
        force_reload=False, onnx=False,
    )
    return model


def _load_wespeaker() -> Any:
    import wespeaker as _ws
    model = _ws.load_model("english")   # ResNet34, 256D — VoxCeleb2 pretrained
    if _device is not None and _device.type == "cuda":
        model.set_gpu(0)
    else:
        model.set_cpu()
    return model


# ── Embedding extraction from raw 16 kHz PCM ─────────────────────────────────

def _extract_embedding(speech_f32: np.ndarray) -> np.ndarray:
    """
    Write accumulated 16 kHz float32 PCM to an in-memory WAV buffer and call
    wespeaker.extract_embedding(). torchaudio.load() supports BytesIO natively,
    which wespeaker uses internally — no temp files, no disk I/O.
    """
    tensor = torch.from_numpy(speech_f32).unsqueeze(0)   # [1, T]
    buf    = io.BytesIO()
    torchaudio.save(buf, tensor, SR_VAD, format="WAV")
    buf.seek(0)
    emb = _speaker_model.extract_embedding(buf)
    if isinstance(emb, torch.Tensor):
        emb = emb.squeeze().cpu().numpy()
    return np.array(emb, dtype=np.float32).flatten()


# ── VAD + Identity worker (thread-pool, never blocks the event loop) ──────────

def _vad_worker(raw_bytes: bytes, frame_id: int, conn_key: str) -> None:

    # 1. Decimate 48 kHz → 16 kHz
    pcm_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
    pcm_f32   = pcm_int16.astype(np.float32) / 32_768.0

    t48 = torch.from_numpy(pcm_f32).unsqueeze(0).to(_device)    # [1, 960]
    t16 = torchaudio.functional.resample(t48, SR_IN, SR_VAD)    # [1, 320]
    chunk_16 = t16.squeeze(0)                                     # [320]

    # 2. Silero VAD — pad to minimum chunk size
    padded = chunk_16
    if chunk_16.shape[0] < SILERO_MIN_CHUNK:
        padded = torch.nn.functional.pad(
            chunk_16, (0, SILERO_MIN_CHUNK - chunk_16.shape[0])
        )

    with torch.no_grad():
        vad_conf: float = _vad_model(padded, SR_VAD).item()

    is_speech = vad_conf > VAD_THRESHOLD

    # 3. Fetch per-connection accumulator
    with _conn_states_lock:
        state = _conn_states.get(conn_key)
    if state is None:
        return   # connection already closed

    # 4. Accumulate / detect segment boundaries
    speech_to_embed: np.ndarray | None = None

    with state.lock:
        if is_speech:
            state.in_speech       = True
            state.silence_counter = 0
            state.speech_16k.append(chunk_16.cpu().numpy())

            # Force-extract if we've hit the 2.0 s segment threshold
            if len(state.speech_16k) >= SPEECH_SEGMENT_FRAMES:
                speech_to_embed  = np.concatenate(state.speech_16k)
                state.speech_16k = []
                # Don't reset in_speech — speaker may still be talking

        else:   # SILENCE frame
            if state.in_speech:
                state.silence_counter += 1
                # Close segment once hangover window elapses
                if state.silence_counter >= SILENCE_HANGOVER:
                    if state.speech_16k:
                        speech_to_embed  = np.concatenate(state.speech_16k)
                    state.speech_16k      = []
                    state.in_speech       = False
                    state.silence_counter = 0

    # 5. Speaker identification — only when a complete segment is ready
    if speech_to_embed is not None and _speaker_model is not None:
        try:
            emb    = _extract_embedding(speech_to_embed)
            result = _library.identify_speaker(emb)
            log.info(
                "[IDENTITY] → Speaker: %s (Conf: %.2f) | frame=%d",
                result.name, result.score, frame_id,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("[IDENTITY] Embedding extraction failed: %s", exc)
    elif is_speech:
        log.info("[SPEECH DETECTED]  conf=%.3f | frame=%d", vad_conf, frame_id)
    else:
        log.info("[SILENCE]          conf=%.3f | frame=%d", vad_conf, frame_id)


# ── FastAPI lifespan ───────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _device, _vad_model, _speaker_model, _library

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", _device)

    log.info("Loading Silero VAD v4 ...")
    _vad_model = await asyncio.to_thread(_load_vad)
    _vad_model = _vad_model.to(_device).eval()
    log.info("✓ Silero VAD v4 ready.")

    log.info("Loading Wespeaker ResNet34 (english, 256D) ...")
    _speaker_model = await asyncio.to_thread(_load_wespeaker)
    log.info("✓ Wespeaker ResNet34 ready on %s.", _device)

    _library = SpeakerLibrary(data_dir=Path(__file__).parent)
    log.info("✓ SpeakerLibrary ready (%d speaker(s)).", len(_library._names))

    yield

    log.info("Amika server shutting down.")


app = FastAPI(
    title="Amika Server — Phase 3: VAD Gate + Speaker Identity",
    lifespan=lifespan,
)


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

    client   = websocket.client
    conn_key = f"{client.host}:{client.port}"
    log.info("Client connected: %s", conn_key)

    with _conn_states_lock:
        _conn_states[conn_key] = ConnState()

    frame_count = 0
    t_start     = time.monotonic()

    try:
        while True:
            data: bytes = await websocket.receive_bytes()
            frame_count += 1

            # ── ECHO FIRST: original 48 kHz chunk, unconditional ──────────────
            await websocket.send_bytes(data)

            # ── Fire VAD + identity as background task (non-blocking) ─────────
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
        log.exception("Error on %s: %s", conn_key, exc)
    finally:
        with _conn_states_lock:
            _conn_states.pop(conn_key, None)
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()


# ── Health probe ──────────────────────────────────────────────────────────────

@app.get("/healthz")
async def healthz() -> dict:
    return {
        "status":            "ok",
        "phase":             3,
        "device":            str(_device),
        "vad_loaded":        _vad_model is not None,
        "wespeaker_loaded":  _speaker_model is not None,
        "enrolled_speakers": len(_library._names) if _library else 0,
    }
