# Architecture D - Commercial Safe - MIT License
import time
import os
import numpy as np

from .models import PipelineOutput, SpeakerMemory
from .state_machine import PipelineStateMachine
from ..database.manager import SpeakerDatabase

class InferenceEngine:
    def __init__(self, bsrnn_path: str = "models/bsrnn.onnx", 
                 ecapa_path: str = "models/ecapa.onnx",
                 db_manager: SpeakerDatabase = None):
        self.bsrnn_path = bsrnn_path
        self.ecapa_path = ecapa_path
        self.db = db_manager
        self.speaker_memory = SpeakerMemory()
        self.state_machine = PipelineStateMachine()
        
        self.bsrnn_session = None
        self.ecapa_session = None
        
        # IO Binding state
        self.io_binding_bsrnn = None
        self.io_binding_ecapa = None

        self._load_models()

    def _load_models(self):
        try:
            import onnxruntime as ort
        except ImportError:
            print("[ENGINE] Warning: onnxruntime not installed.")
            return

        # Maximum speed session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.intra_op_num_threads = 16
        sess_options.inter_op_num_threads = 16
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        
        providers = [
            "CPUExecutionProvider"
        ]

        try:
            if os.path.exists(self.bsrnn_path):
                self.bsrnn_session = ort.InferenceSession(self.bsrnn_path, sess_options, providers=providers)
                print(f"[ENGINE] Loaded {self.bsrnn_path} successfully")
            else:
                print(f"[ENGINE] Warning: {self.bsrnn_path} missing.")

            if os.path.exists(self.ecapa_path):
                self.ecapa_session = ort.InferenceSession(self.ecapa_path, sess_options, providers=providers)
                print(f"[ENGINE] Loaded {self.ecapa_path} successfully")
            else:
                print(f"[ENGINE] Warning: {self.ecapa_path} missing.")
        except Exception as e:
            print(f"[ENGINE] Failed to load models: {e}")

    def warmup(self) -> None:
        """Run 10 dummy inferences to warm up CUDA kernels."""
        if not self.bsrnn_session or not self.ecapa_session:
            return

        print("[ENGINE] Warming up CUDA kernels...")
        t_start = time.perf_counter()
        
        # Dummy inputs (4000 samples @ 16kHz)
        dummy_audio = np.zeros((1, 4000), dtype=np.float32)
        # BSRNN expects float32
        
        for _ in range(10):
            # BSRNN Warmup
            bsrnn_inputs = {self.bsrnn_session.get_inputs()[0].name: dummy_audio}
            bsrnn_out = self.bsrnn_session.run(None, bsrnn_inputs)[0]
            
            # ECAPA Warmup (expects 80-dim logfbank features [batch, frames, 80])
            dummy_ecapa = np.zeros((1, 100, 80), dtype=np.float32)
            ecapa_inputs = {self.ecapa_session.get_inputs()[0].name: dummy_ecapa}
            _ = self.ecapa_session.run(None, ecapa_inputs)
            
        t_end = time.perf_counter()
        print(f"[ENGINE] Warmup complete in {(t_end - t_start)*1000:.2f} ms")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _run_bsrnn(self, audio: np.ndarray) -> tuple:
        """
        Run BSRNN source separation on a mixed audio chunk.

        Returns (stream1, stream2, is_valid) where:
          - stream1/stream2 are [time] float32 arrays
          - is_valid=False signals the caller to fall back to UNIFIED

        Fallback triggers when:
          1. BSRNN session not loaded
          2. ONNX runtime error (logs and returns)
          3. Either stream is silent  (RMS < 0.01)
          4. Both streams too similar (waveform cos-sim > 0.85 = likely same speaker)
        """
        _SILENCE_RMS   = 0.01
        _SIM_THRESHOLD = 0.85

        if not self.bsrnn_session:
            return audio, audio, False

        try:
            input_name = self.bsrnn_session.get_inputs()[0].name
            # BSRNN expects [batch, time]
            input_tensor = np.expand_dims(audio, axis=0).astype(np.float32)  # [1, T]
            outputs = self.bsrnn_session.run(None, {input_name: input_tensor})
            separated = outputs[0]   # [1, 2, T]

            stream1 = separated[0, 0, :]  # [T]
            stream2 = separated[0, 1, :]  # [T]

            # Guard 1 – silence
            rms1 = float(np.sqrt(np.mean(stream1 ** 2)))
            rms2 = float(np.sqrt(np.mean(stream2 ** 2)))
            if rms1 < _SILENCE_RMS:
                print(f"[BSRNN] Stream-1 silent (RMS={rms1:.4f}) → fallback UNIFIED")
                return audio, audio, False
            if rms2 < _SILENCE_RMS:
                print(f"[BSRNN] Stream-2 silent (RMS={rms2:.4f}) → fallback UNIFIED")
                return audio, audio, False

            # Guard 2 – identity (waveform-level cosine similarity)
            wav_sim = self._cosine_similarity(stream1, stream2)
            if wav_sim > _SIM_THRESHOLD:
                print(f"[BSRNN] Streams too similar (wav_cos={wav_sim:.3f}) → fallback UNIFIED")
                return audio, audio, False

            print(f"[BSRNN] OK — RMS1={rms1:.4f} RMS2={rms2:.4f} wav_cos={wav_sim:.3f}")
            return stream1, stream2, True

        except Exception as exc:
            print(f"[BSRNN] Inference error: {exc} — skipping separation")
            return audio, audio, False

    def _run_ecapa_parallel(self, audio1: np.ndarray, audio2: np.ndarray) -> tuple:
        """Run ECAPA on both streams in parallel (simulated via concurrent futures or sequential if no IO binding)."""
        # True CUDA parallel streams require IO Binding on separate streams.
        # Here we do it sequentially for simplicity but ensure IO Binding is utilized if implemented.
        # For Python ONNXRuntime, multi-threading can run them parallel if session is thread safe.
        import concurrent.futures
        
        def run_ecapa(audio):
            if not self.ecapa_session: return np.zeros(192, dtype=np.float32)
            
            from python_speech_features import logfbank
            
            chunk_16k = audio.astype(np.float32)
            # Extract 80-bin Mel-filterbank features
            # logfbank expects raw signal. We use standard 25ms window, 10ms step.
            features = logfbank(chunk_16k, samplerate=16000, nfilt=80, nfft=512)
            
            features = np.expand_dims(features, axis=0).astype(np.float32) # [1, time, 80]
            
            # Use IO Binding to keep on GPU (demonstration of IO binding setup)
            import onnxruntime as ort
            io_binding = self.ecapa_session.io_binding()
            
            device_id = 0
            device_type = 'cpu'
            
            X_ortvalue = ort.OrtValue.ortvalue_from_numpy(features, device_type, device_id)
            io_binding.bind_input(name=self.ecapa_session.get_inputs()[0].name, device_type=device_type, device_id=device_id, element_type=np.float32, shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())
            io_binding.bind_output(self.ecapa_session.get_outputs()[0].name, device_type)
            
            self.ecapa_session.run_with_iobinding(io_binding)
            out = io_binding.get_outputs()[0].numpy()
            return out.flatten()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(run_ecapa, audio1)
            future2 = executor.submit(run_ecapa, audio2)
            emb1 = future1.result()
            emb2 = future2.result()
            
        return emb1, emb2

    def process_chunk(self, audio: np.ndarray) -> PipelineOutput:
        """
        Takes a mono audio chunk (nominally 4000 samples @ 16 kHz).

        UNIFIED path: BSRNN skipped; single ECAPA embedding; one ASR stream.
        SPLIT path  : BSRNN fires; two validated streams; per-speaker ECAPA + ASR.

        The state machine drives which path is active. Fallback to UNIFIED occurs
        whenever BSRNN validation fails (silent stream, identical streams, or error).
        """
        audio = audio.astype(np.float32)
        if len(audio) > 4000:
            audio = audio[:4000]
        elif len(audio) < 4000:
            audio = np.pad(audio, (0, 4000 - len(audio)))

        t_total_start = time.perf_counter()

        # ── 1. Source Separation ──────────────────────────────────────────────
        t_sep_start = time.perf_counter()

        current_state = self.state_machine.state

        if current_state == "SPLIT":
            # Attempt real BSRNN separation; _run_bsrnn() handles all error cases.
            stream1, stream2, bsrnn_valid = self._run_bsrnn(audio)
        else:
            # UNIFIED: skip separation cost entirely.
            stream1, stream2, bsrnn_valid = audio, audio, False

        t_sep_end = time.perf_counter()
        sep_latency = (t_sep_end - t_sep_start) * 1000

        # ── 2. Speaker Identification ─────────────────────────────────────────
        t_id_start = time.perf_counter()

        # Feed BSRNN-separated streams (or identical raw audio in UNIFIED).
        # When BSRNN is valid, emb1/emb2 reflect genuinely different voices;
        # their cosine similarity will be low → state machine can trigger SPLIT.
        emb1, emb2 = self._run_ecapa_parallel(stream1, stream2)
        ecapa_sim   = self._cosine_similarity(emb1, emb2)

        # ── 3. State Machine Update ───────────────────────────────────────────
        self.state_machine.update(audio, stream1, stream2, ecapa_sim)
        # Re-read state after update (may have flipped UNIFIED↔SPLIT)
        new_state = self.state_machine.state

        # The chunk is truly split only if BSRNN was valid AND state confirms SPLIT.
        is_split = bsrnn_valid and (new_state == "SPLIT")

        # ── 4. Identity Matching ──────────────────────────────────────────────
        id1 = self.speaker_memory.match_or_create(emb1)
        id2 = self.speaker_memory.match_or_create(emb2)

        if self.db:
            self.db.upsert_speaker(emb1, id1)
            self.db.upsert_speaker(emb2, id2)

        spk1_obj = self.speaker_memory.known_speakers[id1]
        spk2_obj = self.speaker_memory.known_speakers[id2]

        t_id_end   = time.perf_counter()
        id_latency = (t_id_end - t_id_start) * 1000

        t_total_end   = time.perf_counter()
        total_latency = (t_total_end - t_total_start) * 1000

        # spk1/spk2 audio: separated streams in SPLIT, raw audio in UNIFIED.
        spk1_audio = stream1 if is_split else audio
        spk2_audio = stream2 if is_split else audio

        print(
            f"[ENGINE] sep={sep_latency:.0f}ms id={id_latency:.0f}ms "
            f"total={total_latency:.0f}ms ecapa_sim={ecapa_sim:.3f} "
            f"state={new_state} bsrnn_valid={bsrnn_valid}"
        )

        return PipelineOutput(
            speaker1_audio=spk1_audio,
            speaker2_audio=spk2_audio,
            speaker1_id=spk1_obj,
            speaker2_id=spk2_obj,
            separation_latency_ms=sep_latency,
            id_latency_ms=id_latency,
            total_latency_ms=total_latency,
            is_split=is_split,
            bsrnn_valid=bsrnn_valid,
        )
