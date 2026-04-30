import asyncio
import numpy as np
from .models import BSRNNSeparator, ECAPATDNNManager
from .asr import WhisperASR
from ..database.manager import DatabaseManager

class UnifiedSplit:
    def __init__(self, threshold_db=-40, sample_rate=48000, chunk_size_ms=100):
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

    async def process_loop(self, bridge, audio_state=None):
        self.is_running = True
        bridge.start()
        
        try:
            while self.is_running:
                chunk = bridge.get_latest_chunk()
                
                # Always calculate RMS for monitoring
                rms = self.splitter.calculate_rms(chunk)
                if audio_state:
                    audio_state.current_rms = float(rms)
                
                if self.separator.is_loaded and self.splitter.should_trigger_separation(chunk):
                    # 1. Separate into two speakers
                    speakers = self.separator.separate(chunk)
                    
                    # 2. Identify each speaker
                    for spk_audio in speakers:
                        embedding = self.identity_manager.get_embedding(spk_audio)
                        
                        # Match with database
                        identity = self.match_identity(embedding)
                        
                        # 3. Transcribe
                        # In a real app, we'd save to temp wav or use stream
                        text = self.asr.transcribe_data(spk_audio)
                        
                        # 4. Route/Log
                        print(f"[{identity['name']}] {text}")
                        self.db.update_last_seen(identity['id'])
                
                await asyncio.sleep(0.05)
        finally:
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
