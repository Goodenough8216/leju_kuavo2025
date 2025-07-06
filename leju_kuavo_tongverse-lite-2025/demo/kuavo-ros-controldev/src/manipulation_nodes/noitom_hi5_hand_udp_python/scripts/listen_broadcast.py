import socket
import sys
import threading
import datetime

ctrl_c_pressed = False

def listen_for_broadcasts(port):
    global ctrl_c_pressed
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', port))  # Listen on all interfaces
    sock.settimeout(1)  # Set timeout to 1 second
    print(f"carlos_ Listening for broadcasts on port {port} - Started listening")

    while True:
        try:
            data, addr = sock.recvfrom(1024)
            print(f"carlos_ Received message from broadcast at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {data.decode()} from {addr} on port {port}")
        except socket.timeout:
            if ctrl_c_pressed:
                print(f"carlos_ Thread {port} exiting due to Ctrl+C")
                break
            # print(f"carlos_ Thread {port} timeout. ")
            continue
        except KeyboardInterrupt:
            print(f"carlos_ Thread {port} interrupted. Exiting...")
            break

def main():
    global ctrl_c_pressed
    # Check if a port range is provided as command-line arguments, otherwise use 11000-11010 as default
    start_port = int(sys.argv[1]) if len(sys.argv) > 1 else 11000
    end_port = int(sys.argv[2]) if len(sys.argv) > 2 else 11005

    # Create and start a thread for each port
    threads = []
    for port in range(start_port, end_port + 1):
        thread = threading.Thread(target=listen_for_broadcasts, args=(port,))
        thread.daemon = False  # Set as non-daemon thread to wait for threads to finish
        thread.start()
        threads.append(thread)

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("carlos_ Waiting for all threads to finish...")
        ctrl_c_pressed = True
        for thread in threads:
            thread.join()
        print("carlos_ All threads finished.")

if __name__ == "__main__":
    main()
