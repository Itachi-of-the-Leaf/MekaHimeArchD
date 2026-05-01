import asyncio
import os
import time
import soundfile as sf
import numpy as np
from .models import BSRNNSeparator, ECAPATDNNManager, DeepFilterDenoiser, resample_48k_to_16k
from .asr import WhisperASR
from ..database.manager import DatabaseManager


# ---------------------------------------------------------------------------
# Silence gate
# ---------------------------------------------------------------------------

class UnifiedSplit:
    def __init__(self, threshold_db: float = -40, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.threshold_linear = 10 ** (threshold_db / 20)

    def calculate_rms(self, audio_chunk: np.ndarray) -> float:
        if len(audio_chunk) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio_chunk))))

    def should_trigger_separation(self, audio_chunk: np.ndarray) -> bool:
        return self.calculate_rms(audio_chunk) > self.threshold_linear


# ---------------------------------------------------------------------------
# OLA helper
# ---------------------------------------------------------------------------

def _make_hann_fade(n: int):
    """Return (fade_out, fade_in) Hann-based crossfade windows of length n."""
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    fade_in  = np.sin(t * np.pi / 2) ** 2
    fade_out = 1.0 - fade_in
    return fade_out, fade_in


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class AmikaEngine:
    # ── Processing constants (at 16 kHz, post-separation) ────────────────
    CHUNK_SIZE_48K   = 24_000   # 500 ms of raw capture  @ 48 kHz
    CHUNK_SIZE_16K   = 8_000    # 500 ms of model audio  @ 16 kHz
    FADE_MS          = 25       # crossfade window (ms)

    def __init__(self, db_path: str = "data/ears.db"):
        self.splitter         = UnifiedSplit()
        self.denoiser         = DeepFilterDenoiser()
        self.separator        = BSRNNSeparator()
        self.identity_manager = ECAPATDNNManager()
        self.asr              = WhisperASR()
        self.db               = DatabaseManager(db_path)
        self.is_running       = False

        # OLA state at 16 kHz
        self._fade_samples_16k = int((self.FADE_MS / 1000) * 16_000)   # 400 samples
        self._fade_out_16k, self._fade_in_16k = _make_hann_fade(self._fade_samples_16k)
        self._prev_tails_16k = [
            np.zeros(self._fade_samples_16k, dtype=np.float32),
            np.zeros(self._fade_samples_16k, dtype=np.float32),
        ]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def process_loop(self, bridge, audio_state=None):
        self.is_running = True
        bridge.start()

        os.makedirs("outputs", exist_ok=True)
        f_spk1 = sf.SoundFile("outputs/speaker_1.wav", mode="w", samplerate=16000, channels=1)
        f_spk2 = sf.SoundFile("outputs/speaker_2.wav", mode="w", samplerate=16000, channels=1)

        audio_accumulator = np.empty(0, dtype=np.float32)

        try:
            while self.is_running:
                # ── 1. Drain the bridge ───────────────────────────────
                new_chunk = bridge.get_latest_chunk()
                if new_chunk is not None and len(new_chunk) > 0:
                    audio_accumulator = np.concatenate((audio_accumulator, new_chunk))

                if len(audio_accumulator) < self.CHUNK_SIZE_48K:
                    await asyncio.sleep(0.01)
                    continue

                # ── 2. Slice exactly 500 ms @ 48 kHz ─────────────────
                chunk_48k         = audio_accumulator[:self.CHUNK_SIZE_48K]
                audio_accumulator = audio_accumulator[self.CHUNK_SIZE_48K:]

                # ── Denoise at native 48kHz first! ─────────────────────
                if self.denoiser.is_loaded:
                    chunk_48k = self.denoiser.denoise(chunk_48k)

                # ── 3. RMS monitoring ─────────────────────────────────
                rms = self.splitter.calculate_rms(chunk_48k)
                if audio_state:
                    audio_state.current_rms = rms

                if not self.separator.is_loaded:
                    await asyncio.sleep(0.01)
                    continue

                # ── 4. Separate (resampling happens inside BSRNNSeparator) ──
                t0 = time.perf_counter()
                spk1, spk2 = self.separator.separate(chunk_48k)
                print(f"[PERF] separation: {time.perf_counter() - t0:.3f}s  "
                      f"spk1={spk1.shape}  spk2={spk2.shape}")

                # ── 5. Permutation resolution via ECAPA embeddings ────
                #
                # We compute embeddings on the *raw* separated streams before
                # any OLA smoothing so the embedding model sees unmodified
                # speaker characteristics.
                if self.identity_manager.is_loaded:
                    emb_a = self.identity_manager.get_embedding(spk1)
                    emb_b = self.identity_manager.get_embedding(spk2)
                    if self.identity_manager.resolve_permutation(emb_a, emb_b):
                        spk1, spk2 = spk2, spk1   # correct the flip

                # ── 6. OLA crossfade + write ──────────────────────────
                speakers     = [spk1, spk2]
                smoothed     = []
                fade_n       = self._fade_samples_16k
                fade_out     = self._fade_out_16k
                fade_in      = self._fade_in_16k

                for i, spk_audio in enumerate(speakers):
                    if len(spk_audio) <= fade_n:
                        # Chunk too short to crossfade — write as-is
                        smoothed.append(spk_audio)
                        continue

                    out = spk_audio.copy()

                    # Blend the opening of this chunk with the stored tail of
                    # the previous chunk using a Hann-based crossfade.
                    out[:fade_n] = (
                        self._prev_tails_16k[i] * fade_out
                        + out[:fade_n]           * fade_in
                    )

                    # Store the tail (last fade_n samples) for next iteration.
                    # We write the FULL chunk including the tail so no audio is
                    # lost — the crossfade only affects the *boundary* region,
                    # not the bulk of the signal.
                    self._prev_tails_16k[i] = spk_audio[-fade_n:].copy()

                    smoothed.append(out)

                f_spk1.write(smoothed[0])
                f_spk2.write(smoothed[1])

                # ── 7. Identity logging ───────────────────────────────
                if self.identity_manager.is_loaded:
                    for lane_idx, spk_audio in enumerate(smoothed):
                        embedding = self.identity_manager.get_embedding(spk_audio)
                        if embedding is not None:
                            identity = self._match_identity(embedding)
                            self.db.update_last_seen(identity["id"])

                await asyncio.sleep(0.001)

        finally:
            # ── Flush remaining OLA tails so no audio is silently dropped ──
            for i, tail in enumerate(self._prev_tails_16k):
                if np.any(tail != 0):
                    fade_n   = self._fade_samples_16k
                    fade_out = self._fade_out_16k
                    tail_out = tail * fade_out   # fade the dangling tail to zero
                    if i == 0:
                        f_spk1.write(tail_out)
                    else:
                        f_spk2.write(tail_out)

            f_spk1.close()
            f_spk2.close()
            bridge.stop()
            print("[INFO] AmikaEngine stopped. Output files closed.")

    # ------------------------------------------------------------------
    # Identity matching
    # ------------------------------------------------------------------

    def _match_identity(self, embedding: np.ndarray) -> dict:
        known = self.db.get_all_identities()
        best_match, min_dist = None, float("inf")
        for identity in known:
            dist = float(np.linalg.norm(identity["embedding"] - embedding))
            if dist < min_dist:
                min_dist, best_match = dist, identity

        if best_match and min_dist < 0.5:
            return best_match

        new_id = self.db.add_identity(embedding, name="Unknown", priority=2)
        return {"id": new_id, "name": "Unknown", "priority": 2}

    # kept for API compatibility
    def match_identity(self, embedding: np.ndarray) -> dict:
        return self._match_identity(embedding)

    def stop(self):
        self.is_running = False