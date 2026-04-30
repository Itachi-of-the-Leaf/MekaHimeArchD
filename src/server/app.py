import asyncio
import json
from granian import Granian
from granian.constants import Interfaces

class App:
    def __init__(self, scope):
        self.scope = scope

    async def __call__(self, receive, send):
        if self.scope['type'] == 'websocket':
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
            'body': b'Amika\'s Ears API Active'
        })

    async def handle_websocket(self, receive, send):
        # WebSocket handshake
        while True:
            message = await receive()
            if message['type'] == 'websocket.connect':
                await send({'type': 'websocket.accept'})
            elif message['type'] == 'websocket.receive':
                # Handle incoming commands
                data = json.loads(message.get('text', '{}'))
                # Echo or process
                await send({
                    'type': 'websocket.send',
                    'text': json.dumps({"status": "received", "data": data})
                })
            elif message['type'] == 'websocket.disconnect':
                break

def main():
    server = Granian(
        "src.server.app:App",
        address="0.0.0.0",
        port=8000,
        interface=Interfaces.ASGI,
        threads=4
    )
    server.serve()

if __name__ == "__main__":
    main()
