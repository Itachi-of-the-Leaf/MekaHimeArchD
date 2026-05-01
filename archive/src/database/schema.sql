CREATE TABLE IF NOT EXISTS identities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT DEFAULT 'Unknown',
    embedding BLOB NOT NULL,
    priority INTEGER DEFAULT 2, -- 0: Master, 1: Friend, 2: Unknown
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audio_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    identity_id INTEGER,
    transcript TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(identity_id) REFERENCES identities(id)
);
