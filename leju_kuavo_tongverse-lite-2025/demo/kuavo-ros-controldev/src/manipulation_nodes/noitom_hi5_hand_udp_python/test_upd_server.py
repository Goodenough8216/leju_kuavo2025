import socket
import datetime

def start_udp_server():
    server_ip = "0.0.0.0"
    server_port = 8080
    buffer_size = 1024

    # Create a datagram socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind the socket to the server address
    sock.bind((server_ip, server_port))
    print(f"UDP server up and listening on {server_port}")

    try:
        while True:
            # Wait for a message from the client
            message, address = sock.recvfrom(buffer_size)


            print(f"{datetime.datetime.now()} - Received message: {message.decode()} from {address}")


            # Send a reply to the client
            reply = "Hello from carlos server"
            sock.sendto(reply.encode(), address)
    except KeyboardInterrupt:
        print("UDP server is shutting down.")
    finally:
        sock.close()

if __name__ == "__main__":
    start_udp_server()
