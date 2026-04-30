import subprocess
import os

class WhisperASR:
    def __init__(self, model_path="models/ggml-base.en.bin", executable_path="./bin/whisper-cli"):
        self.model_path = model_path
        self.executable_path = executable_path

    def transcribe(self, audio_path):
        # Using the CLI as a fallback if binding isn't available
        # In production, we'd use a shared library/binding for speed
        cmd = [
            self.executable_path,
            "-m", self.model_path,
            "-f", audio_path,
            "-nt" # no timestamps for faster output
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout.strip()

    async def transcribe_async(self, audio_data):
        # Placeholder for real-time streaming ASR
        # whisper.cpp supports stream, but needs a specific implementation
        pass
