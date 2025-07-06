import asyncio
import websockets
from datetime import datetime
import threading
from webrtc_singaling_server import WebRTCSinglingServer 

if __name__ == "__main__":
    server = WebRTCSinglingServer()
    server.start()  # Start the server in a background thread

    # Your other code can run here
    try:
        while True:
            # Get connected clients from the server
            connected_clients = server.get_connected_clients()
            print("Connected clients:", connected_clients)
            print("Main thread is running...")
            asyncio.get_event_loop().run_until_complete(asyncio.sleep(5))
    except KeyboardInterrupt:
        server.stop()
