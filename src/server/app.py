from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from src.bridge import audio_bridge
from src.core.engine import AmikaEngine
import threading
import asyncio
import json

class AudioState:
    def __init__(self, bridge):
        self.bridge = bridge
        self.current_rms = 0.0

    def get_rms(self):
        return self.current_rms

def start_engine(engine, bridge, audio_state):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(engine.process_loop(bridge, audio_state))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Boot C++ Bridge INSIDE the worker process
    print("Worker Booting: Initializing C++ Audio Bridge...")
    bridge = audio_bridge.AudioBridge(sampleRate=48000, channels=1, bufferSize=24000)
    engine = AmikaEngine()
    audio_state = AudioState(bridge)
    app.state.audio = audio_state
    
    # Start Engine
    engine_thread = threading.Thread(target=start_engine, args=(engine, bridge, audio_state), daemon=True)
    engine_thread.start()
    
    yield # Application runs here
    
    # Teardown C++ Bridge gracefully when worker shuts down
    print("Worker Shutting Down: Cleaning up C++ Bridge...")
    if hasattr(bridge, 'stop'):
        bridge.stop()
    del bridge

def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    
    @app.get("/")
    async def root():
        return "Amika's Ears: Granian Server Active"
        
    @app.websocket("/")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                audio_state = app.state.audio
                rms = audio_state.get_rms() if audio_state else 0.0
                await websocket.send_text(json.dumps({
                    "type": "rms_update",
                    "value": rms
                }))
                await asyncio.sleep(0.033)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"WebSocket error: {e}")

    return app
