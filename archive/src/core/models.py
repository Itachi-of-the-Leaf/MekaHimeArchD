import onnxruntime as ort
import numpy as np
import torch
import torch.nn.functional as F
import scipy.signal
from speechbrain.inference.speaker import EncoderClassifier

# DeepFilterNet 0.5.6 imports removed torchaudio modules. We must mock them before importing df.
import sys, types
if "torchaudio.backend" not in sys.modules:
    m = types.ModuleType("torchaudio.backend")
    m.common = types.ModuleType("torchaudio.backend.common")
    m.common.AudioMetaData = type("AudioMetaData", (), {})
    sys.modules["torchaudio.backend"] = m
    sys.modules["torchaudio.backend.common"] = m.common

try:
    from df.enhance import enhance, init_df
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resample_48k_to_16k(chunk_f32: np.ndarray) -> np.ndarray:
    """
    Proper anti-aliased downsample from 48 kHz → 16 kHz using a polyphase FIR
    filter (scipy.signal.resample_poly).

    DO NOT use naive slicing (chunk[::3]). That skips the anti-aliasing filter,
    causing aliasing artefacts that corrupt the signal before it even reaches
    the separation or embedding models.
    """
    # resample_poly expects a 1-D array
    flat = chunk_f32.ravel()
    out = scipy.signal.resample_poly(flat, up=1, down=3).astype(np.float32)
    return out


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D or 2-D (1, D) embedding vectors."""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# DeepFilterDenoiser
# ---------------------------------------------------------------------------

class DeepFilterDenoiser:
    def __init__(self):
        self.is_loaded = False
        try:
            # init_df automatically loads the default DeepFilterNet3 model
            self.model, self.df_state, _ = init_df()
            print("[INFO] DeepFilterDenoiser loaded successfully.")
            self.is_loaded = True
        except Exception as e:
            print(f"[WARNING] DeepFilterDenoiser could not load: {e}")

    def denoise(self, audio_chunk_48k: np.ndarray) -> np.ndarray:
        if not self.is_loaded:
            return audio_chunk_48k
            
        chunk_f32 = audio_chunk_48k.astype(np.float32)
        peak = np.max(np.abs(chunk_f32))
        if peak > 1.0:
            chunk_f32 = chunk_f32 / 32768.0

        # df expects [C, T] tensor
        audio_tensor = torch.from_numpy(chunk_f32).unsqueeze(0)
        
        # enhance applies DeepFilterNet in real-time
        clean_tensor = enhance(self.model, self.df_state, audio_tensor, pad=True)
        
        # Return cleaned (T,) ndarray
        return clean_tensor.squeeze(0).numpy()


# ---------------------------------------------------------------------------
# BSRNNSeparator
# ---------------------------------------------------------------------------

class BSRNNSeparator:
    def __init__(self, model_path: str = "models/bsrnn.onnx"):
        self.is_loaded = False
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            active = self.session.get_providers()[0]
            print(f"[INFO] BSRNNSeparator: using {active} ({model_path})")
            self.is_loaded = True
        except Exception as e:
            print(f"[WARNING] BSRNN model not found at '{model_path}'. "
                  "Running in passthrough mode. ({e})")

    def separate(self, audio_chunk_48k: np.ndarray):
        """
        Accept a raw 48 kHz float32 chunk, resample to 16 kHz with proper
        anti-aliasing, run BSRNN, and return (spk1_16k, spk2_16k).

        Returns (None, None) if the model is not loaded.
        """
        if not self.is_loaded:
            return None, None

        chunk_f32 = audio_chunk_48k.astype(np.float32)

        # Normalise if int-range values somehow slipped through
        peak = np.max(np.abs(chunk_f32))
        if peak > 1.0:
            chunk_f32 = chunk_f32 / 32768.0

        # ── FIX: proper anti-aliased resample (was chunk_f32[::3]) ──────────
        chunk_16k = resample_48k_to_16k(chunk_f32)
        # ─────────────────────────────────────────────────────────────────────

        # BSRNN expects shape [1, T]
        model_input = chunk_16k[np.newaxis, :]

        inputs = {self.session.get_inputs()[0].name: model_input}
        outputs = self.session.run(None, inputs)
        out_array = outputs[0]

        print(f"[DEBUG] BSRNN output shape: {out_array.shape}")

        # Robust shape handling
        if out_array.ndim == 3:
            if out_array.shape[2] == 2:      # [1, T, 2]
                spk1 = out_array[0, :, 0]
                spk2 = out_array[0, :, 1]
            elif out_array.shape[1] == 2:    # [1, 2, T]
                spk1 = out_array[0, 0, :]
                spk2 = out_array[0, 1, :]
            else:
                raise ValueError(f"Unexpected 3-D BSRNN output shape: {out_array.shape}")
        elif out_array.ndim == 2:            # [2, T]
            spk1 = out_array[0, :]
            spk2 = out_array[1, :]
        else:
            raise ValueError(f"Unexpected BSRNN output shape: {out_array.shape}")

        return spk1.astype(np.float32), spk2.astype(np.float32)


# ---------------------------------------------------------------------------
# ECAPATDNNManager
# ---------------------------------------------------------------------------

class ECAPATDNNManager:
    """
    Speaker embedding manager.

    Two responsibilities:
      1. get_embedding(audio_16k)  →  np.ndarray  (embedding vector)
      2. resolve_permutation(emb_a, emb_b, ref0, ref1)  →  bool (swap?)

    Permutation resolution uses cosine similarity against per-speaker EMA
    references, which is semantically correct and robust to silence / low-
    energy regions (unlike raw-waveform MAE comparisons).
    """

    def __init__(self,
                 model_path: str = "models/ecapa.onnx",
                 ema_alpha: float = 0.1):
        self.is_loaded = False
        self.ema_alpha = ema_alpha

        # Per-speaker EMA embedding references  {lane_index: np.ndarray}
        self.lane_refs: dict[int, np.ndarray] = {}

        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            active = self.session.get_providers()[0]
            print(f"[INFO] ECAPATDNNManager: using {active} ({model_path})")

            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": "cpu"},
            )
            self.is_loaded = True
        except Exception as e:
            print(f"[WARNING] ECAPA model not found at '{model_path}'. "
                  f"Identity / permutation resolution disabled. ({e})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_embedding(self, audio_16k: np.ndarray) -> np.ndarray | None:
        """
        Compute an ECAPA-TDNN embedding for a 16 kHz mono audio chunk.
        Input may come from either the bridge (still 48 kHz) or the separator
        output (already 16 kHz). Pass 16 kHz audio here — resample upstream.
        """
        if not self.is_loaded:
            return None

        chunk_f32 = audio_16k.ravel().astype(np.float32)
        peak = np.max(np.abs(chunk_f32))
        if peak > 1.0:
            chunk_f32 = chunk_f32 / 32768.0

        # Pad / trim to exactly 16 000 samples (1 second) for stable features
        target = 16000
        if len(chunk_f32) < target:
            chunk_f32 = np.pad(chunk_f32, (0, target - len(chunk_f32)))
        else:
            chunk_f32 = chunk_f32[:target]

        wavs = torch.from_numpy(chunk_f32[np.newaxis, :])   # [1, T]
        wav_lens = torch.ones(1)

        with torch.no_grad():
            feats = self.classifier.mods.compute_features(wavs)
            feats = self.classifier.mods.mean_var_norm(feats, wav_lens)

        inputs = {self.session.get_inputs()[0].name: feats.numpy()}
        outputs = self.session.run(None, inputs)
        return outputs[0].astype(np.float32)   # shape [1, D] or [D]

    def resolve_permutation(
        self,
        emb_a: np.ndarray,
        emb_b: np.ndarray,
    ) -> bool:
        """
        Given embeddings for lane-0 (emb_a) and lane-1 (emb_b) of the current
        chunk, decide whether the lanes need to be swapped to maintain speaker
        consistency with the previous chunk.

        Returns True  → swap (spk1, spk2 = spk2, spk1)
        Returns False → no change

        On the very first call (no references yet) always returns False and
        seeds the references.
        """
        if not self.is_loaded or emb_a is None or emb_b is None:
            return False

        if 0 not in self.lane_refs:
            # First chunk — seed references, no swap decision possible
            self.lane_refs[0] = emb_a.ravel().copy()
            self.lane_refs[1] = emb_b.ravel().copy()
            return False

        ref0 = self.lane_refs[0]
        ref1 = self.lane_refs[1]

        # Straight assignment: lane-0 → speaker-0, lane-1 → speaker-1
        score_straight = cosine_similarity(emb_a, ref0) + cosine_similarity(emb_b, ref1)
        # Flipped assignment: lane-0 → speaker-1, lane-1 → speaker-0
        score_flipped  = cosine_similarity(emb_a, ref1) + cosine_similarity(emb_b, ref0)

        swap = score_flipped > score_straight
        if swap:
            print("[DEBUG] Lane flip detected via embedding similarity — correcting.")

        # Update EMA references with the winning assignment
        winner_a = emb_b if swap else emb_a   # what we'll call speaker-0
        winner_b = emb_a if swap else emb_b   # what we'll call speaker-1
        self._update_ema(0, winner_a)
        self._update_ema(1, winner_b)

        return swap

    def update_identity(self, identity_id: int, new_embedding: np.ndarray):
        """Legacy per-identity EMA update used by the database layer."""
        emb = new_embedding.ravel().astype(np.float32)
        if identity_id not in self.lane_refs:
            self.lane_refs[identity_id] = emb
        else:
            alpha = self.ema_alpha
            self.lane_refs[identity_id] = (1 - alpha) * self.lane_refs[identity_id] + alpha * emb
        return self.lane_refs[identity_id]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_ema(self, lane: int, new_emb: np.ndarray):
        e = new_emb.ravel().astype(np.float32)
        if lane not in self.lane_refs:
            self.lane_refs[lane] = e
        else:
            alpha = self.ema_alpha
            self.lane_refs[lane] = (1 - alpha) * self.lane_refs[lane] + alpha * e