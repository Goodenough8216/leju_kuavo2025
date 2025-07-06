#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

text=$1

if [ -z "$text" ]; then
    echo "Error: No text provided."
    exit 1
fi

if [ -f "/opt/lejurobot/kuavo-wifi-announce/venv/bin/activate" ]; then
    echo "playing text: $text"
    source /opt/lejurobot/kuavo-wifi-announce/venv/bin/activate
    python3 $SCRIPT_DIR/kuavo_wifi_announce.py --play-text "$text"
else
    echo -e "\033[31m❌ => The \033[33m'kuavo-wifi-announce'\033[0m is not installed. Please install it first.\033[0m"
    exit 1
fi