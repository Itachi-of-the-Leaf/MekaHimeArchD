# Architecture D - Commercial Safe - MIT License
import sqlite3
import numpy as np
import os
import time

from ..core.models import SpeakerEmbedding

class SpeakerDatabase:
    def __init__(self, db_path="data/ears.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.init_db()

    def init_db(self):
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        if os.path.exists(schema_path):
            with open(schema_path, "r") as f:
                self.conn.executescript(f.read())
            self.conn.commit()

    def upsert_speaker(self, embedding: np.ndarray, speaker_id: str, name: str = None) -> str:
        embedding_blob = embedding.astype(np.float32).tobytes()
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT id FROM speakers WHERE id = ?", (speaker_id,))
        if cursor.fetchone():
            cursor.execute(
                "UPDATE speakers SET embedding_blob = ?, last_seen = CURRENT_TIMESTAMP, encounter_count = encounter_count + 1 WHERE id = ?",
                (embedding_blob, speaker_id)
            )
        else:
            cursor.execute(
                "INSERT INTO speakers (id, name, embedding_blob) VALUES (?, ?, ?)",
                (speaker_id, name, embedding_blob)
            )
            
        self.conn.commit()
        return speaker_id

    def find_speaker(self, embedding: np.ndarray, threshold: float = 0.75) -> str | None:
        speakers = self.get_all_speakers()
        best_match_id = None
        highest_sim = -1.0

        for spk in speakers:
            dot_product = np.dot(embedding, spk.embedding)
            norm_a = np.linalg.norm(embedding)
            norm_b = np.linalg.norm(spk.embedding)
            sim = dot_product / (norm_a * norm_b)

            if sim > highest_sim:
                highest_sim = sim
                best_match_id = spk.id

        if best_match_id and highest_sim >= threshold:
            return best_match_id
        return None

    def get_all_speakers(self) -> list[SpeakerEmbedding]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, embedding_blob, last_seen FROM speakers")
        rows = cursor.fetchall()
        
        speakers = []
        for row in rows:
            speakers.append(SpeakerEmbedding(
                id=row[0],
                embedding=np.frombuffer(row[2], dtype=np.float32),
                confidence=1.0,
                last_seen=time.mktime(time.strptime(row[3], "%Y-%m-%d %H:%M:%S")) if isinstance(row[3], str) else row[3]
            ))
        return speakers
