import asyncio
import threading
from src.bridge import audio_bridge
from src.core.engine import AmikaEngine
from src.server.app import AudioState, run_server

def start_engine(engine, bridge):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(engine.process_loop(bridge))

def main():
    # 1. Initialize Audio Bridge (48kHz, Mono)
    # The bridge handles its own thread for miniaudio capture
    print("Initializing Audio Bridge...")
    bridge = audio_bridge.AudioBridge(sampleRate=48000, channels=1, bufferSize=4800)
    
    # 2. Initialize Engine
    engine = AmikaEngine()
    
    # 3. Shared State for Server
    audio_state = AudioState(bridge)
    
    # 4. Start Engine in a separate thread
    engine_thread = threading.Thread(target=start_engine, args=(engine, bridge), daemon=True)
    engine_thread.start()
    
    # 5. Start Granian Server (Blocking)
    print("Starting Granian Server on http://0.0.0.0:8000")
    run_server(audio_state)

if __name__ == "__main__":
    main()
