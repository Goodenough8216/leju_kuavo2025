#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

REPORT_ROBOT_NETWORK_INFO_SERVICE=$SCRIPT_DIR/report_robot_network_info.service
REPORT_ROBOT_NETWORK_INFO_SERVICE_SCRIPT=$SCRIPT_DIR/RRNIS_script.py
REPORT_ROBOT_NETWORK_INFO_ENV=$SCRIPT_DIR/RRNIS.env

pip3 install requests
sudo chmod +x $REPORT_ROBOT_NETWORK_INFO_SERVICE_SCRIPT
sudo cp $REPORT_ROBOT_NETWORK_INFO_SERVICE_SCRIPT /usr/local/bin/RRNIS_script

sudo mkdir -p /etc/environment.d
sudo cp $REPORT_ROBOT_NETWORK_INFO_ENV /etc/environment.d/RRNIS.env

sudo cp $REPORT_ROBOT_NETWORK_INFO_SERVICE /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable report_robot_network_info.service