# Architecture D - Commercial Safe - MIT License
from dataclasses import dataclass, field
import numpy as np
import time

@dataclass
class SpeakerEmbedding:
    id: str
    embedding: np.ndarray
    confidence: float
    last_seen: float = field(default_factory=time.time)

@dataclass
class PipelineOutput:
    speaker1_audio: np.ndarray
    speaker2_audio: np.ndarray
    speaker1_id: SpeakerEmbedding
    speaker2_id: SpeakerEmbedding
    separation_latency_ms: float
    id_latency_ms: float
    total_latency_ms: float
    is_split: bool = False      # True only when BSRNN fired AND streams are distinct
    bsrnn_valid: bool = False   # True when BSRNN produced non-silent, non-identical streams

class SpeakerMemory:
    def __init__(self, alpha: float = 0.1):
        self.known_speakers: dict[str, SpeakerEmbedding] = {}
        self.alpha = alpha

    def match_or_create(self, embedding: np.ndarray, threshold: float = 0.75) -> str:
        best_match_id = None
        highest_sim = -1.0

        for spk_id, spk_data in self.known_speakers.items():
            # Cosine similarity
            dot_product = np.dot(embedding, spk_data.embedding)
            norm_a = np.linalg.norm(embedding)
            norm_b = np.linalg.norm(spk_data.embedding)
            if norm_a == 0 or norm_b == 0:
                sim = 0.0
            else:
                sim = dot_product / (norm_a * norm_b)

            if sim > highest_sim:
                highest_sim = sim
                best_match_id = spk_id

        if best_match_id and highest_sim >= threshold:
            # Update EMA
            current_emb = self.known_speakers[best_match_id].embedding
            updated_emb = (1 - self.alpha) * current_emb + self.alpha * embedding
            self.known_speakers[best_match_id].embedding = updated_emb
            self.known_speakers[best_match_id].last_seen = time.time()
            return best_match_id
        
        # Create new
        import uuid
        new_id = str(uuid.uuid4())
        self.known_speakers[new_id] = SpeakerEmbedding(
            id=new_id,
            embedding=embedding,
            confidence=1.0,
            last_seen=time.time()
        )
        return new_id
