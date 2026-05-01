"""
server/echo_server.py — Phase 1: "Dumb" Echo Pipe

A raw, asynchronous FastAPI WebSocket server. No AI. No PyTorch.
When it receives a binary audio chunk it immediately echoes it back.

TCP_NODELAY is set on the underlying asyncio transport to disable
Nagle's algorithm and ensure each 20ms frame is flushed instantly.

Usage (local):
    uvicorn echo_server:app --host 0.0.0.0 --port 8765

Usage (Colab — see README.md):
    !uvicorn echo_server:app --host 0.0.0.0 --port 8765 &
    # then run cloudflared in a second cell
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("amika.echo")

app = FastAPI(title="Amika Echo Server — Phase 1")


def _set_tcp_nodelay(websocket: WebSocket) -> None:
    """
    Reach into the asyncio transport and disable Nagle's algorithm.
    This is the lowest-level hook available without patching uvicorn.
    """
    try:
        transport: Any = websocket.scope.get("transport")  # set by some ASGI servers
        if transport is None:
            # Starlette / uvicorn path: the transport lives on the underlying
            # h11 connection object.
            transport = getattr(
                websocket._receive.__self__,  # type: ignore[attr-defined]
                "_transport",
                None,
            )
        if transport is not None:
            sock = transport.get_extra_info("socket")
            if sock is not None:
                import socket
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                log.debug("TCP_NODELAY set on accepted socket.")
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not set TCP_NODELAY: %s", exc)


@app.websocket("/audio")
async def audio_echo(websocket: WebSocket) -> None:
    await websocket.accept()
    _set_tcp_nodelay(websocket)

    client = websocket.client
    log.info("Client connected: %s:%s", client.host, client.port)

    frame_count = 0
    t_start = time.monotonic()

    try:
        while True:
            # Receive a binary frame (20ms = 1920 bytes of L16 PCM @ 48kHz mono)
            data: bytes = await websocket.receive_bytes()

            # ── ECHO — this is all Phase 1 does ───────────────────────────────
            await websocket.send_bytes(data)

            frame_count += 1

            # Telemetry: log every 50 frames (~1 second of audio)
            if frame_count % 50 == 0:
                elapsed = time.monotonic() - t_start
                log.info(
                    "frames=%d  elapsed=%.1fs  avg_fps=%.1f  frame_bytes=%d",
                    frame_count,
                    elapsed,
                    frame_count / elapsed,
                    len(data),
                )

    except WebSocketDisconnect:
        log.info("Client disconnected after %d frames.", frame_count)
    except Exception as exc:  # noqa: BLE001
        log.exception("Unexpected error: %s", exc)
    finally:
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()


@app.get("/healthz")
async def healthz() -> dict:
    """Simple health-check endpoint for Cloudflare Tunnel probes."""
    return {"status": "ok", "phase": 1}
