#!/bin/bash

echo -e "\033[32m\n🚀🚀🚀 开始卸载...\n\033[0m"

if systemctl list-units --full -all | grep -Fq 'create_ap.service'; then
  sudo systemctl disable create_ap
fi

if dpkg -l | grep -q linux-wifi-hotspot; then
  sudo apt remove -y linux-wifi-hotspot
fi

echo -e "\033[32m\n🚀🚀🚀 卸载成功...\n\033[0m"
