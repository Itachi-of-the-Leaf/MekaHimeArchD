"""
Pure-Python AudioBridge fallback (Architecture D).

Used when the C++ nanobind bridge is not compiled. Provides the same
interface as the C++ AudioBridge class using sounddevice for capture.

Includes latency logging for diagnostics.
"""

import time
from typing import Optional

import numpy as np
import sounddevice as sd


class AudioBridge:
    """Software audio capture bridge using sounddevice (PortAudio)."""

    def __init__(
        self,
        sampleRate: int = 48000,
        channels: int = 1,
        bufferSize: int = 48000,
    ) -> None:
        self.sampleRate: int = sampleRate
        self.channels: int = channels
        self.buffer: list[float] = []
        self._last_chunk_time: Optional[float] = None
        self._chunk_count: int = 0

        self.stream: sd.InputStream = sd.InputStream(
            samplerate=sampleRate,
            channels=channels,
            callback=self._callback,
        )
        print(f"[AudioBridge] Fallback initialized (sounddevice) @ {sampleRate}Hz, {channels}ch")

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            print(f"[AudioBridge] Stream error: {status}")
        # Flatten and store — zero-copy via .flatten() on the copy
        self.buffer.extend(indata.copy().flatten())

    def start(self) -> None:
        if not self.stream.active:
            self.stream.start()
            self._last_chunk_time = time.perf_counter()
            print("[AudioBridge] Stream started")

    def stop(self) -> None:
        if self.stream.active:
            self.stream.stop()
            print(f"[AudioBridge] Stream stopped (served {self._chunk_count} chunks)")

    def get_latest_chunk(self) -> np.ndarray:
        now = time.perf_counter()

        if not self.buffer:
            return np.array([], dtype=np.float32)

        # Convert buffer to numpy array
        chunk = np.array(self.buffer, dtype=np.float32)
        self.buffer = []  # Clear for next call
        self._chunk_count += 1

        # Latency logging
        if self._last_chunk_time is not None:
            delta_ms = (now - self._last_chunk_time) * 1000.0
            duration_ms = (len(chunk) / self.sampleRate) * 1000.0
            if self._chunk_count % 20 == 0:  # Log every 20th chunk to avoid spam
                print(
                    f"[AudioBridge] Chunk #{self._chunk_count}: "
                    f"{len(chunk)} samples ({duration_ms:.1f}ms audio), "
                    f"interval={delta_ms:.1f}ms"
                )
        self._last_chunk_time = now

        return chunk

    def get_sample_rate(self) -> int:
        """Return the configured sample rate."""
        return self.sampleRate
