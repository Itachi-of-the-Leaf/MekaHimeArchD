# Architecture D - Commercial Safe - MIT License
import os
import sys
import time
import signal
import numpy as np

# Adjust imports to match the new structure
from src.core.engine import InferenceEngine
from src.core.vad import VADEngine
from src.core.asr import ASREngine
from src.core.state_machine import PipelineStateMachine
from src.database.manager import SpeakerDatabase

def main() -> None:
    print("--- MekaHimeArchD: Full Speed-Optimized Pipeline ---")
    t_start = time.perf_counter()

    # 1. Initialize Database
    db = SpeakerDatabase(db_path="data/ears.db")

    # 2. Initialize Engines
    engine = InferenceEngine(db_manager=db)
    vad = VADEngine()
    asr = ASREngine()
    
    # 3. Warmup
    # Warmup disabled on CPU - adds 52s delay with no benefit
    # engine.warmup()

    # Audio Setup
    try:
        import sounddevice as sd
    except ImportError:
        print("[ERROR] sounddevice is required for audio capture.")
        print("Install with: uv add sounddevice")
        return

    sample_rate = 16000
    chunk_ms = 250
    chunk_samples = int(sample_rate * (chunk_ms / 1000.0)) # 4000
    
    print(f"[AUDIO] Starting capture: {sample_rate}Hz, {chunk_ms}ms chunks ({chunk_samples} samples)")

    is_running = True
    def signal_handler(sig, frame):
        nonlocal is_running
        print("\n[INFO] Graceful shutdown requested...")
        is_running = False

    signal.signal(signal.SIGINT, signal_handler)

    audio_queue = []
    
    # Pre-allocate variables for statistics
    total_chunks = 0
    total_latency = 0.0

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[AUDIO] Status: {status}")
        # Flatten to 1D, apply 5x digital gain, and clip
        amplified = np.clip(indata[:, 0] * 5.0, -1.0, 1.0)
        audio_queue.append(amplified)

    stream = sd.InputStream(samplerate=sample_rate, channels=1, blocksize=chunk_samples, callback=audio_callback)
    
    with stream:
        print("[INFO] Pipeline active. Listening...")
        while is_running:
            if not audio_queue:
                time.sleep(0.01)
                continue

            chunk = audio_queue.pop(0)
            if len(chunk) < chunk_samples:
                continue

            loop_start = time.perf_counter()

            # VAD Check
            rms_val = np.sqrt(np.mean(chunk ** 2))
            print(f"[AUDIO] Chunk RMS: {rms_val:.4f}")
            if vad.is_speech(chunk):
                # Process audio chunk
                out = engine.process_chunk(chunk)
                
                # ASR (VAD-gated, run based on state)
                # If unified, transcribe raw chunk. If split, might want to transcribe separated.
                # For simplicity, we just transcribe the original chunk here or spk1.
                # Prompt: "In UNIFIED mode: pass audio directly to ASR, skip BSRNN"
                # (Note: engine currently runs BSRNN anyway, but we transcribe based on state logic if needed)
                
                state = engine.state_machine.state

                if not out.is_split:
                    # UNIFIED: one speaker or BSRNN fallback — transcribe once
                    transcript = asr.transcribe(chunk)
                    log_prefix = f"[UNIFIED | spk:{out.speaker1_id.id[:8]}]"
                else:
                    # SPLIT: BSRNN produced two valid, distinct streams
                    # Transcribe each separated stream independently
                    transcript1 = asr.transcribe(out.speaker1_audio)
                    transcript2 = asr.transcribe(out.speaker2_audio)
                    s1 = out.speaker1_id.id[:8]
                    s2 = out.speaker2_id.id[:8]
                    transcript = (
                        f"\n  ┌─ Spk-A [{s1}]: {transcript1 or '(silence)'}"
                        f"\n  └─ Spk-B [{s2}]: {transcript2 or '(silence)'}"
                    )
                    log_prefix = f"[SPLIT | {s1} & {s2}]"

                loop_latency_ms = (time.perf_counter() - loop_start) * 1000
                total_latency += loop_latency_ms
                total_chunks += 1

                print(f"{log_prefix} Latency: {loop_latency_ms:.1f}ms {transcript}")
            else:
                pass # Silence, do nothing

    # Shutdown
    asr.shutdown()
    
    print("\n--- Pipeline Statistics ---")
    if total_chunks > 0:
        print(f"Total Chunks Processed: {total_chunks}")
        print(f"Average Pipeline Latency: {total_latency / total_chunks:.2f}ms")
    else:
        print("No speech detected during session.")

if __name__ == "__main__":
    main()
