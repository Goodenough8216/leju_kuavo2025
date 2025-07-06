import socket
# Import the hand_pose_pb2 module using a relative path
import sys
import os

# Add the parent directory to the system path to allow relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the hand_pose_pb2 module
import protos.hand_pose_pb2 as event_pb2

import time

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Define the server address and port
server_address = ('localhost', 10000)

# Create an Event message
event = event_pb2.LejuHandPoseEvent()

import time

for _ in range(100):
    try:
        # Update timestamp before sending
        event.timestamp = int(time.time() * 1000)  # Convert to milliseconds
        
        # Serialize the message to a string
        serialized_data = event.SerializeToString()
        
        # Send data
        print(f'Sending timestamp: {event.timestamp}')
        sent = sock.sendto(serialized_data, server_address)
        time.sleep(0.0001)  # Delay of 10ms
    except Exception as e:
        print(f'An error occurred: {e}')

print('Closing socket')
sock.close()
