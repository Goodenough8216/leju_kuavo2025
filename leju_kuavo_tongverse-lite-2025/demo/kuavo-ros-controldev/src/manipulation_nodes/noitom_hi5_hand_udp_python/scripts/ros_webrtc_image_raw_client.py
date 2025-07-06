import cv2
import asyncio
import json
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCIceCandidate
from aiortc.contrib.signaling import BYE
import websockets
from aiortc.contrib.media import MediaStreamTrack
from datetime import datetime
from av import VideoFrame
import av
import fractions
import sys
import subprocess
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import signal
from webrtcvideostreamclient import WebRTCVideoStreamClient


async def main():
    if len(sys.argv) < 3:
        print("Usage: python client.py <server_ip> <ros_topic>")
        sys.exit(-1)

    server_ip = sys.argv[1]
    ros_topic = sys.argv[2]

    client = WebRTCVideoStreamClient(server_ip, ros_topic)
    client.start()

    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("Ctrl-C pressed, exiting...")
        loop.close()
