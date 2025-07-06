import socket
import sys
import os
# Add the parent directory to the system path to allow relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import protos.hand_pose_pb2 as event_pb2

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind the socket to the port
server_address = ('localhost', 10000)
sock.bind(server_address)

print('Waiting to receive message')

while True:
    data, address = sock.recvfrom(4096)
    
    # Deserialize the message
    event = event_pb2.LejuHandPoseEvent()
    event.ParseFromString(data)
    
    print(f'Received timestamp: {event.timestamp}')
