import sqlite3
import numpy as np
import os

class DatabaseManager:
    def __init__(self, db_path="data/ears.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.init_db()

    def init_db(self):
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        with open(schema_path, "r") as f:
            self.conn.executescript(f.read())
        self.conn.commit()

    def add_identity(self, embedding, name="Unknown", priority=2):
        # Convert numpy embedding to blob
        embedding_blob = embedding.astype(np.float32).tobytes()
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO identities (name, embedding, priority) VALUES (?, ?, ?)",
            (name, embedding_blob, priority)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_all_identities(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, embedding, priority FROM identities ORDER BY priority ASC")
        rows = cursor.fetchall()
        
        identities = []
        for row in rows:
            identities.append({
                "id": row[0],
                "name": row[1],
                "embedding": np.frombuffer(row[2], dtype=np.float32),
                "priority": row[3]
            })
        return identities

    def update_last_seen(self, identity_id):
        self.conn.execute(
            "UPDATE identities SET last_seen = CURRENT_TIMESTAMP WHERE id = ?",
            (identity_id,)
        )
        self.conn.commit()
