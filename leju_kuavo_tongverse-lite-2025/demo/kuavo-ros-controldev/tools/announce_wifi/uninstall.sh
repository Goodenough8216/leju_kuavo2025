#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
INSTALL_DIR=/opt/lejurobot/
APP_NAME=kuavo-wifi-announce

#@@@ ROOT!
if [ $(id -u) != "0" ]; then
  echo -e "\033[31m\nYou must be root to run this script, please use root to uninstall!\n\033[0m"
  exit 1
fi

#@@@ REMOVE-SERVICE!
if [ -f "/etc/systemd/system/kuavo-wifi-announce.service" ]; then
  systemctl stop kuavo-wifi-announce.service
  systemctl disable kuavo-wifi-announce.service
fi

connection_exists() {
    nmcli connection show | grep -q "$1"
}

if connection_exists "kuavo-hotspot"; then
  echo -e "\033[33mDelete connection AP(Hotspot): "kuavo-hotspot" \033[0m"
  nmcli connection delete "kuavo-hotspot"
fi

rm -rf "/etc/systemd/system/kuavo-wifi-announce.service"
systemctl daemon-reload

#@@@ REMOVE-BIN!
rm -rf "$INSTALL_DIR/$APP_NAME"

#@@@ END!
echo -e "\033[32m\n👋👋👋 Uninstall Success! '$APP_NAME' was removed, Goodbye!\n \033[0m"
