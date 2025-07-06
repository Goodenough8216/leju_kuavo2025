#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source /opt/lejurobot/kuavo-wifi-announce/venv/bin/activate
python3 $SCRIPT_DIR/kuavo_wifi_announce.py