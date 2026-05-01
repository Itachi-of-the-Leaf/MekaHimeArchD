import asyncio
import os
import soundfile as sf
import numpy as np
from .models import BSRNNSeparator, ECAPATDNNManager
from .asr import WhisperASR
from ..database.manager import DatabaseManager

class UnifiedSplit:
    def __init__(self, threshold_db=-40, sample_rate=48000, chunk_size_ms=500):
        self.threshold_db = threshold_db
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * (chunk_size_ms / 1000))
        self.threshold_linear = 10 ** (threshold_db / 20)

    def calculate_rms(self, audio_chunk):
        if len(audio_chunk) == 0:
            return 0
        return np.sqrt(np.mean(np.square(audio_chunk)))

    def should_trigger_separation(self, audio_chunk):
        rms = self.calculate_rms(audio_chunk)
        return rms > self.threshold_linear

class AmikaEngine:
    def __init__(self, db_path="data/ears.db"):
        self.splitter = UnifiedSplit()
        self.separator = BSRNNSeparator()
        self.identity_manager = ECAPATDNNManager()
        self.asr = WhisperASR()
        self.db = DatabaseManager(db_path)
        self.is_running = False
        
        # OLA Initialization
        self.fade_ms = 25
        self.fade_samples = int((self.fade_ms / 1000) * self.splitter.sample_rate)
        self.fade_out = np.linspace(1.0, 0.0, self.fade_samples, dtype=np.float32)
        self.fade_in = np.linspace(0.0, 1.0, self.fade_samples, dtype=np.float32)
        self.prev_tails = [np.zeros(self.fade_samples, dtype=np.float32), 
                           np.zeros(self.fade_samples, dtype=np.float32)]

    async def process_loop(self, bridge, audio_state=None):
        self.is_running = True
        bridge.start()
        
        # Resampled OLA setup (16kHz)
        fade_ms = 25
        fade_samples_16k = int((fade_ms / 1000) * 16000)
        fade_out_16k = np.linspace(1.0, 0.0, fade_samples_16k, dtype=np.float32)
        fade_in_16k = np.linspace(0.0, 1.0, fade_samples_16k, dtype=np.float32)
        prev_tails_16k = [np.zeros(fade_samples_16k, dtype=np.float32), 
                           np.zeros(fade_samples_16k, dtype=np.float32)]
        
        # Lane Alignment Setup
        prev_spk1_tail = None

        os.makedirs("outputs", exist_ok=True)
        f_spk1 = sf.SoundFile('outputs/speaker_1.wav', mode='w', samplerate=16000, channels=1)
        f_spk2 = sf.SoundFile('outputs/speaker_2.wav', mode='w', samplerate=16000, channels=1)
        
        import time
        audio_accumulator = np.array([], dtype=np.float32)
        chunk_size_samples = 24000 # 500ms @ 48kHz
        
        try:
            while self.is_running:
                # 1. Accumulate audio from the bridge
                new_chunk = bridge.get_latest_chunk()
                if new_chunk is not None and len(new_chunk) > 0:
                    audio_accumulator = np.concatenate((audio_accumulator, new_chunk))
                
                # 2. Process only when we have a full 24000 samples
                if len(audio_accumulator) < chunk_size_samples:
                    await asyncio.sleep(0.05)
                    continue
                
                # Slice exactly 24000 samples
                chunk = audio_accumulator[:chunk_size_samples]
                # Keep the remainder for the next loop
                audio_accumulator = audio_accumulator[chunk_size_samples:]
                
                inference_start = time.time()
                
                # Always calculate RMS for monitoring
                rms = self.splitter.calculate_rms(chunk)
                if audio_state:
                    audio_state.current_rms = float(rms)
                
                if self.separator.is_loaded:
                    # 1. Separate into two speakers
                    spk1, spk2 = self.separator.separate(chunk) 
                    inference_end = time.time()
                    print(f"Processing time: {inference_end - inference_start:.3f} seconds")

                    # Lane Alignment Correction
                    if prev_spk1_tail is not None:
                        # Extract the first 500 samples of the incoming chunks
                        head1 = spk1[:500]
                        head2 = spk2[:500]
                        
                        # Calculate MAE
                        error_straight = np.mean(np.abs(prev_spk1_tail - head1))
                        error_flipped = np.mean(np.abs(prev_spk1_tail - head2))
                        
                        # If the flipped error is smaller, the neural network flipped the lanes
                        if error_flipped < error_straight:
                            print("[DEBUG] Lane Flip Detected! Correcting permutation.")
                            spk1, spk2 = spk2, spk1 # Swap them back
                    
                    # Update tail for next iteration
                    prev_spk1_tail = spk1[-500:]
                    
                    speakers = [spk1, spk2]
                    
                    smoothed_speakers = []
                    for i, spk_audio in enumerate(speakers):
                        # Note: spk_audio is resampled 16kHz
                        if len(spk_audio) > fade_samples_16k:
                            spk_audio_copy = spk_audio.copy()
                            # Crossfade boundary with previous tail
                            spk_audio_copy[:fade_samples_16k] = (prev_tails_16k[i] * fade_out_16k) + (spk_audio_copy[:fade_samples_16k] * fade_in_16k)
                            # Store current tail for next iteration
                            prev_tails_16k[i] = spk_audio[-fade_samples_16k:].copy()
                            # Output excludes the new tail to prevent premature overlap writing
                            out_audio = spk_audio_copy[:-fade_samples_16k]
                        else:
                            out_audio = spk_audio
                        
                        smoothed_speakers.append(out_audio)
                        
                        if i == 0:
                            f_spk1.write(out_audio)
                        elif i == 1:
                            f_spk2.write(out_audio)
                            
                    # 2. Identify and log
                    for spk_audio in smoothed_speakers:
                        embedding = self.identity_manager.get_embedding(spk_audio)
                        if embedding is not None:
                            identity = self.match_identity(embedding)
                            self.db.update_last_seen(identity['id'])
                
                await asyncio.sleep(0.01)
        finally:
            f_spk1.close()
            f_spk2.close()
            bridge.stop()

    def match_identity(self, embedding):
        known_identities = self.db.get_all_identities()
        best_match = None
        min_dist = float('inf')
        
        for identity in known_identities:
            dist = np.linalg.norm(identity['embedding'] - embedding)
            if dist < min_dist:
                min_dist = dist
                best_match = identity
        
        # Threshold for recognition
        if best_match and min_dist < 0.5:
            return best_match
        else:
            # Create new temporary identity
            new_id = self.db.add_identity(embedding, name="Unknown", priority=2)
            return {"id": new_id, "name": "Unknown", "priority": 2}

    def stop(self):
        self.is_running = False
