# Architecture D - Commercial Safe - MIT License
import numpy as np
import collections

class VADEngine:
    def __init__(self, threshold: float = 0.15, context_chunks: int = 3):
        """
        Uses Silero VAD. To maintain the 'NO torch imports in src/' constraint,
        we use the ONNX version of Silero VAD.
        """
        self.threshold = threshold
        self.context_chunks = context_chunks
        self.buffer = collections.deque(maxlen=context_chunks)
        self.session = None

        try:
            import onnxruntime as ort
            # Requires silero_vad.onnx to be in models/
            model_path = "models/silero_vad.onnx"
            import os
            if os.path.exists(model_path):
                self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            else:
                print(f"[VAD] Warning: {model_path} not found. VAD will fallback to basic RMS energy.")
        except ImportError:
            print("[VAD] Warning: onnxruntime not found. VAD will fallback to basic RMS energy.")
        
        # Setup Silero internal states if ONNX is loaded
        if self.session:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def _rms_energy(self, chunk: np.ndarray) -> float:
        if len(chunk) == 0: return 0.0
        return float(np.sqrt(np.mean(np.square(chunk))))

    def get_speech_probability(self, audio_chunk: np.ndarray) -> float:
        self.buffer.append(audio_chunk)

        if not self.session:
            # Fallback RMS heuristic
            rms = self._rms_energy(audio_chunk)
            # Rough mapping: > 0.01 is speech
            prob = min(1.0, rms * 100) 
            return prob

        chunk_f32 = audio_chunk.astype(np.float32)
        if np.max(np.abs(chunk_f32)) > 1.0:
            chunk_f32 /= 32768.0
            
        sr_tensor = np.array(16000, dtype=np.int64)
        probs = []
        
        for i in range(0, len(chunk_f32) - 512 + 1, 512):
            tensor_chunk = np.expand_dims(chunk_f32[i:i+512], axis=0)
            inputs = {
                'input': tensor_chunk,
                'sr': sr_tensor,
                'state': self._state
            }
            try:
                out, state = self.session.run(None, inputs)
                self._state = state
                probs.append(float(out[0][0]))
            except Exception as e:
                print(f"[VAD] Error running inference: {e}")
                
        probability = max(probs) if probs else self._rms_energy(audio_chunk) * 100
        rms = self._rms_energy(audio_chunk)
        is_speech = probability > self.threshold
        print(f"[VAD] RMS: {rms:.4f} | Speech prob: {probability:.4f} | Speech: {is_speech}")
        return probability

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        rms = self._rms_energy(audio_chunk)
        if rms > 0.008:
            return True
            
        prob = self.get_speech_probability(audio_chunk)
        return prob > self.threshold
