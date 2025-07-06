import socket
import sys
import struct
import time
import datetime
import math

bone_names = [
    "ForeArm",
    "Hand",
    "HandThumb1",
    "HandThumb2",
    "HandThumb3",
    "InHandIndex",
    "HandIndex1",
    "HandIndex2",
    "HandIndex3",
    "InHandMiddle",
    "HandMiddle1",
    "HandMiddle2",
    "HandMiddle3",
    "InHandRing",
    "HandRing1",
    "HandRing2",
    "HandRing3",
    "InHandPinky",
    "HandPinky1",
    "HandPinky2",
    "HandPinky3"
]

class Point3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Point3D(x={self.x:8}, y={self.y:8}, z={self.z:8})"

class Quaternion:
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __repr__(self):
        return f"Quat (x={self.x:6}, y={self.y:6}, z={self.z:6}, w={self.w:6})"

def euler_to_quaternion(euler_angles):
    # Convert Euler angles (in degrees) to radians
    roll = math.radians(euler_angles.x)
    pitch = math.radians(euler_angles.y)
    yaw = math.radians(euler_angles.z)

    # Abbreviations for the various angular functions
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    # Calculate the Quaternion components
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return Quaternion(round(x,3), round(y,3), round(z,3), round(w,3))



def connect_to_udp_server():
    server_ip = "10.10.20.45"
    server_port = 9019

    if len(sys.argv) > 1:
        server_port = int(sys.argv[1])

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Set a timeout for the socket operations
    timeout_seconds = 0.5  # Set the timeout to 20 seconds
    sock.settimeout(timeout_seconds)

    try:
        while True:  # Use a loop to send and receive data continuously
            # Send a test message
            sys.stdout.write("\033[2J\033[H")  # ANSI escape code to clear the screen and move cursor to top-left
            message = "Hello UDP Server!"
            try:
                sock.sendto(message.encode(), (server_ip, server_port))
            except socket.timeout:
                print(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')} Timed out while trying to send data to server.")
                continue  # Skip the rest of the loop and try again

            # sock.sendto(message.encode(), (server_ip, server_port))
            # print(f"Sent message: {message}")

            # Receive response
            try:
                data, server = sock.recvfrom(4096)
            except socket.timeout:
                print(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')} Timed out waiting for data from server.")
                continue  # Skip the rest of the loop and try again
            print(f"Data length is {len(data)}")


            float_array = []
            points = []

            # Convert every 4 bytes to a float and add to the float_array
            for i in range(0, len(data), 4):
                float_value = struct.unpack('<f', data[i:i+4])[0]
                float_array.append(float_value)

            # Convert float_array to a structured array of Point3D objects
            for i in range(0, len(float_array), 3):
                if i + 2 < len(float_array):
                    point = Point3D(float_array[i], float_array[i+1], float_array[i+2])
                    points.append(point)

            # Print the structured array of points


            if server_port == 9019:
                hand_side = "left hand <-"
            else:
                hand_side = "right hand ->"

            print(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')}, {hand_side} recv data length {len(data)}")

            for index, point in enumerate(points):
                print(f"{bone_names[index]:12} {euler_to_quaternion(point)}")

            # print("Connection successful")
            time.sleep(0.05)  # Pause for 1 second before the next iteration[1][2][3][4][5][6]
            # time.sleep(1)

    except Exception as e:
        print(f"Connection failed: {e}")
    finally:
        sock.close()

# Execute the function to test the connection
connect_to_udp_server()
