# Architecture D - Commercial Safe - MIT License
import numpy as np
import threading
import queue
import time

class ASREngine:
    def __init__(self, model_size: str = "base"):
        self.transcription_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_running = True
        self.model = None
        
        try:
            from faster_whisper import WhisperModel
            # Using CPU for ASR to save GPU memory for audio pipeline, or GPU if small enough
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            print(f"[ASR] Loaded whisper-{model_size} via faster-whisper")
        except ImportError:
            print("[ASR] Warning: faster-whisper not installed. Transcriptions will be empty.")
            print("[ASR] Install with: uv add faster-whisper")

        # Start async worker
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _worker_loop(self):
        while self.is_running:
            try:
                audio = self.transcription_queue.get(timeout=1.0)
                if audio is None: 
                    break

                if self.model:
                    # faster-whisper expects float32 in [-1, 1] at 16kHz
                    audio_f32 = audio.astype(np.float32)
                    if np.max(np.abs(audio_f32)) > 1.0:
                        audio_f32 /= 32768.0

                    segments, info = self.model.transcribe(audio_f32, beam_size=1)
                    text = " ".join([segment.text for segment in segments]).strip()
                else:
                    time.sleep(0.1) # Simulate some work
                    text = ""

                self.result_queue.put(text)
                self.transcription_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ASR] Worker error: {e}")
                self.result_queue.put("")

    def transcribe(self, audio: np.ndarray) -> str:
        """
        VAD-gated method. Only call this when speech is detected.
        This blocks until transcription is complete for the chunk.
        """
        if len(audio) == 0:
            return ""
            
        self.transcription_queue.put(audio)
        try:
            # Wait for result
            result = self.result_queue.get(timeout=5.0)
            return result
        except queue.Empty:
            return ""

    def shutdown(self):
        self.is_running = False
        self.transcription_queue.put(None)
        self.worker_thread.join(timeout=2.0)
