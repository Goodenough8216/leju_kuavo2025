#!/usr/bin/env python3
import rospy
import os

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from utils import get_wifi_ip, get_wifi, get_mac_address
import json
import websockets
import asyncio
import socket
import signal
import argparse
from queue import Empty
from typing import Set

from utils import get_wifi_ip, get_wifi, get_mac_address
from handler import (
    response_queue,
    websocket_message_handler,
    Response,
    cleanup_websocket,
    set_robot_type,
    init_ros_node,
)

from kuavo_ros_interfaces.msg import planArmState

# Set to store active WebSocket connections
active_connections: Set[websockets.WebSocketServerProtocol] = set()

ROBOT_NAME = os.getenv("ROBOT_NAME", "KUAVO")
ROBOT_IP = get_wifi_ip()
ROBOT_CONNECT_WIFI = get_wifi()
ROBOT_WS_ADDRESS = f"ws://{ROBOT_IP}:8888"
BROADCAST_IP = f"{ROBOT_IP.rsplit('.', 1)[0]}.255"
BROADCAST_PORT = 8443
ROBOT_MAC_ADDRESS = get_mac_address()
ROBOT_USERNAME = "lab"
ROBOT_ACTION_FILE_FOLDER = "~/.config/lejuconfig/action_files"

robot_info = {
    "data": {
        "robot_name": ROBOT_NAME,
        "robot_ip": ROBOT_IP,
        "robot_connect_wifi": ROBOT_CONNECT_WIFI,
        "robot_ws_address": ROBOT_WS_ADDRESS,
        "robot_action_file_folder": ROBOT_ACTION_FILE_FOLDER,
        "robot_username": ROBOT_USERNAME,
        "robot_mac_address": ROBOT_MAC_ADDRESS,
    }
}

async def broadcast_robot_info():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print(f"Broadcasting to {BROADCAST_IP}:{BROADCAST_PORT}")
    print(f"Broadcasting robot info: {robot_info}")
    while True:
        message = json.dumps(robot_info).encode("utf-8")
        sock.sendto(message, (BROADCAST_IP, BROADCAST_PORT))
        await asyncio.sleep(1)

async def handle_websocket(websocket, path):
    try:
        active_connections.add(websocket)
        print(f"Client connected: {websocket.remote_address}")
        async for message in websocket:
            print(f"Received message from client: {message}")
            data = json.loads(message)
            await websocket_message_handler(websocket, data)
    except websockets.exceptions.ConnectionClosed:
        print(f"Connection closed for client: {websocket.remote_address}")
    finally:
        active_connections.remove(websocket)
        cleanup_websocket(websocket)
        print(f"Client disconnected: {websocket.remote_address}")

async def send_to_websockets(response: Response):
    payload = json.dumps(response.payload.__dict__)
    target = response.target
    if target == "all":
        print(f"Broadcasting message to all clients: {payload}")
        await asyncio.gather(
            *[connection.send(payload) for connection in active_connections],
            return_exceptions=True,
        )
    else:
        # print(f"Sending message to specific client: {payload}")
        if target in active_connections:
            try:
                await target.send(payload)
            except websockets.exceptions.ConnectionClosed:
                print(f"Connection closed for client: {target.remote_address}")
                active_connections.remove(target)
                cleanup_websocket(target)
        else:
            print(f"Client {target} not found in active connections")

async def process_responses():
    print("Starting to process responses")
    last_sent_message = None
    while True:
        try:
            response: Response = await asyncio.get_event_loop().run_in_executor(None, response_queue.get, True, 0.1)
            current_message = json.dumps(response.payload.__dict__)
            
            if current_message != last_sent_message:
                await send_to_websockets(response)
                last_sent_message = current_message
            else:
                print("Skipped sending duplicate message")
        except Empty:
            await asyncio.sleep(0.001)

async def websocket_server():
    server = await websockets.serve(handle_websocket, "0.0.0.0", 8888)
    print("WebSocket server started on ws://0.0.0.0:8888")
    await server.wait_closed()


async def main(robot_type):
    set_robot_type(robot_type)
    print("Starting ROS node initialization")
    ros_init_task = asyncio.create_task(init_ros_node())

    print("Starting broadcast task")
    broadcast_task = asyncio.create_task(broadcast_robot_info())
    print("Starting WebSocket server task")
    websocket_server_task = asyncio.create_task(websocket_server())
    print("Starting response processing task")
    process_responses_task = asyncio.create_task(process_responses())

    # Wait for ROS node initialization to complete
    await ros_init_task

    # Start ROS spin task after initialization

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    print("All tasks started, waiting for stop event")
    try:
        await stop_event.wait()
    finally:
        print("Shutting down gracefully...")
        broadcast_task.cancel()
        websocket_server_task.cancel()
        process_responses_task.cancel()
        try:
            await asyncio.gather(
                broadcast_task,
                websocket_server_task,
                process_responses_task,
                return_exceptions=True
            )
        except asyncio.CancelledError:
            pass
        print("Shutdown complete")

def parse_args():
    parser = argparse.ArgumentParser(description='Plan arm action websocket server')
    parser.add_argument('--robot_type', type=str, default='kuavo', help='Robot type')
    
    args, unknown = parser.parse_known_args()
    
    return args

if __name__ == "__main__":
    args = parse_args()

    asyncio.run(main(args.robot_type))
