import asyncio
import json
import numpy as np
from granian import Granian
from granian.constants import Interfaces

class AudioState:
    def __init__(self, bridge):
        self.bridge = bridge

    def get_rms(self):
        chunk = self.bridge.get_latest_chunk()
        if len(chunk) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(chunk))))

class GranianServer:
    def __init__(self, audio_state):
        self.audio_state = audio_state

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'websocket':
            await self.handle_websocket(receive, send)
        else:
            await self.handle_http(receive, send)

    async def handle_http(self, receive, send):
        await send({
            'type': 'http.response.start',
            'status': 200,
            'headers': [(b'content-type', b'text/plain')]
        })
        await send({
            'type': 'http.response.body',
            'body': b"Amika's Ears: Granian Server Active"
        })

    async def handle_websocket(self, receive, send):
        await send({'type': 'websocket.accept'})
        
        # Streaming loop
        try:
            while True:
                # Check for disconnect
                # (A real implementation would use a more robust way to check receive())
                
                # Calculate RMS from the bridge
                rms = self.audio_state.get_rms()
                
                # Send to frontend
                await send({
                    'type': 'websocket.send',
                    'text': json.dumps({
                        "type": "rms_update",
                        "value": rms
                    })
                })
                
                # ~30 FPS for the dashboard
                await asyncio.sleep(0.033)
                
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            # Handle disconnect logic if needed
            pass

def run_server(audio_state, host="0.0.0.0", port=8000):
    server = Granian(
        target=GranianServer(audio_state),
        address=host,
        port=port,
        interface=Interfaces.ASGI,
        threads=2
    )
    server.serve()
