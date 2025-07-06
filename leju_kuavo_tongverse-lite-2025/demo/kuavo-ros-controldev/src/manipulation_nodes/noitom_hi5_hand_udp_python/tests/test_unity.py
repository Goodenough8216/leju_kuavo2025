import socket
import sys
import os
import time

from pprint import pprint

# Add the parent directory to the system path to allow relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the hand_pose_pb2 module
import protos.hand_pose_pb2 as event_pb2

# Check if server address and port are provided as a single command-line argument
if len(sys.argv) != 2:
    raise Exception("Usage: python test_unity.py <server_address:port>")

# Parse the server address and port
try:
    server_address, port = sys.argv[1].split(':')
    port = int(port)
except ValueError:
    raise Exception("Argument must be in the format <server_address:port> and port must be an integer")

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Define the server address and port
server_address = (server_address, port)

# Create an Event message
event = event_pb2.LejuHandPoseEvent()

# Send "hi" to the server first
sock.sendto(b'hi', server_address)

for _ in range(100):
    try:
        # Wait for data
        data, _ = sock.recvfrom(4096)

        # Deserialize the message from the received data
        event.ParseFromString(data)

        # Print the timestamp
        print(f'Received timestamp: {event.timestamp}')
        # pprint(event.poses)
        time.sleep(0.0005)  # Delay of 10ms
    except Exception as e:
        print(f'An error occurred: {e}')

print('Closing socket')
sock.close()
