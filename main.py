import asyncio
import signal
from src.core.engine import AmikaEngine
from src.bridge import audio_bridge # Requires compilation

async def main():
    # 1. Initialize Engine
    engine = AmikaEngine()
    
    # 2. Initialize Bridge (48kHz, Mono, 100ms chunk)
    bridge = audio_bridge.AudioBridge(sampleRate=48000, channels=1, bufferSize=4800)
    
    # 3. Handle Shutdown
    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()
    
    def shutdown():
        print("\nShutting down Amika's Ears...")
        engine.stop()
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown)

    # 4. Start Processing
    print("Amika's Ears started. Listening...")
    await engine.process_loop(bridge)

if __name__ == "__main__":
    asyncio.run(main())
