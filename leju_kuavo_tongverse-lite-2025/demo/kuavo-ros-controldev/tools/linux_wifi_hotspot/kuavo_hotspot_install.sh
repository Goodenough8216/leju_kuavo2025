#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
HOTSPOT_PASSWORD="kuavo123456"
HOTSPOT_SSID="kuavo-$(hostname)的热点"

echo -e "\033[32m\n🚀🚀🚀 开始安装...\n\033[0m"

# @@@ INSTALL
sudo add-apt-repository ppa:lakinduakash/lwh -y || { echo -e "\e[31m❌ Failed to add repository\e[0m"; exit 1; }
sudo apt update
sudo apt install -y linux-wifi-hotspot || { echo -e "\e[31m❌ Failed to install linux-wifi-hotspot\e[0m"; exit 1; }

# @@@ CREATE HOTSPOT
wlan0=$(iw dev | awk '$1=="Interface" && $2 !~ /^(ap|lo|docker|veth)/{print $2; exit}')
supported_ap_mode=$(iw list | grep -A 20 'Supported interface modes' | grep '* AP')
if [ -n "$supported_ap_mode" ] && [ -n "$wlan0" ]; then
  echo -e "\033[32m\n✅✅✅ 无线网卡名称: $wlan0, 支持热点模式! \n \033[0m"
else
  echo -e "\033[33m🚫🚫🚫 安装失败, 你的机器人的无线网卡似乎不支持AP模式, 请检查你的网卡型号!\033[0m"
  exit 1
fi

# Read ROBOT_SERIAL_NUMBER from environment file and set it as HOTSPOT_SSID
if [ -f "/etc/environment.d/RRNIS.env" ]; then
    HOTSPOT_SSID=$(grep "ROBOT_SERIAL_NUMBER" /etc/environment.d/RRNIS.env | cut -d'=' -f2)
else
    read -p $'\e[32m🤖🤖🤖 请输入你的机器人序列号(例如:MT-5)来生成你的热点名称:\e[0m' ROBOT_SERIAL_NUMBER; echo "ROBOT_SERIAL_NUMBER=$ROBOT_SERIAL_NUMBER" | sudo tee /etc/environment.d/RRNIS.env > /dev/null; HOTSPOT_SSID="${ROBOT_SERIAL_NUMBER}"
fi

# Loop to check if HOTSPOT_SSID exceeds 24 characters, if so, prompt user to re-enter
max_length=22
while [ ${#HOTSPOT_SSID} -gt $max_length ]; do
    echo -e "\033[31m❌ 热点名称不能超过 $max_length 个字符，请重新输入。\033[0m"
    read -p $'\e[32m🤖🤖🤖 请输入你的机器人序列号(例如:MT-5)来生成你的热点名称:\e[0m' -e ROBOT_SERIAL_NUMBER
    HOTSPOT_SSID="${ROBOT_SERIAL_NUMBER}"
done

HOTSPOT_SSID="${HOTSPOT_SSID}的热点"
echo -e ""

sudo create_ap $wlan0 $wlan0 "$HOTSPOT_SSID" "$HOTSPOT_PASSWORD" --mkconfig /etc/create_ap.conf --freq-band 2.4 > /dev/null 2>&1
if [ $? -eq 0 ]; then
  echo -e "\033[32m🎉🎉🎉 热点创建成功, 热点名称: $HOTSPOT_SSID, 密码:$HOTSPOT_PASSWORD \n \033[0m"
else
  echo -e "\033[31m❌❌❌ 热点创建失败!\033[0m"
  exit 1
fi

# @@@ AUTO START
sudo systemctl enable create_ap
sudo systemctl start create_ap
if [ "$(systemctl is-active create_ap)" = "active" ]; then
  echo -e "\033[32m🚀🚀🚀 热点服务已启动! \n \033[0m"
fi
