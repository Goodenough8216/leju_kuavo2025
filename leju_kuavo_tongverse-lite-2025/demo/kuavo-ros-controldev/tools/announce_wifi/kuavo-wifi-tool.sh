#!/bin/bash

INSTALL_DIR=/opt/lejurobot/
APP_NAME=kuavo-wifi-announce

##################
# @ Help Functions
##################

function ip_zh_cn_text() {
    local ip=$1
    local digits=("零" "一" "二" "三" "四" "五" "六" "七" "八" "九")
    local result=""

    IFS='.' read -r -a octets <<< "$ip"
    for octet in "${octets[@]}"; do
        for (( i=0; i<${#octet}; i++ )); do
            digit=${octet:$i:1}
            result+="${digits[$digit]} "
        done
        result+="点 "
    done

    # remove last `点`
    result=${result%点 }

    # remove last space
    result=${result% }

    echo "$result"
}

function usage() {
    echo -e "\033[33m\nUsage: $0 [option]"
    echo ""
    echo "Options:"
    echo "  -l, --show-wifi      Show available Wi-Fi networks"
    echo "  -c, --connect-wifi   Connect to a Wi-Fi network"
    echo "                       Usage: $0 -c wifi \"wifi-name\" password [\"wifi-password\"]"
    echo "  -h, --help           Display this help message"
    echo -e "\033[0m"
}

function play_text_audio() {
    local tts_text=$1
     if [ -f "$INSTALL_DIR/$APP_NAME/venv/bin/activate" ]; then
        source $INSTALL_DIR/$APP_NAME/venv/bin/activate
        python3 $INSTALL_DIR/$APP_NAME/bin/kuavo_wifi_announce.py --play-text "$tts_text"
        deactivate
     else
        echo -e "\033[31m❌ => The \033[33m'kuavo-wifi-announce'\033[0m is not installed. Please install it first.\033[0m"
        exit 1
     fi
}

function tips_wifi_connected () {
    if [ -f "$INSTALL_DIR/$APP_NAME/venv/bin/activate" ]; then
        local max_attempts=10
        local attempt=0
        local uuid=""
        local wifi_name=""
        local ip_address=""

        while [ $attempt -lt $max_attempts ]; do
            uuid=$(nmcli -t -f TYPE,DEVICE,UUID connection show --active | grep '802-11-wireless' | cut -d':' -f3)
            if [[ -n "$uuid" ]]; then
                wifi_name=$(nmcli -t -f 802-11-wireless.ssid con show  $uuid|cut -d':' -f2)
                ip_address=$(nmcli -t -f IP4.ADDRESS con show  $uuid|cut -d':' -f2| cut -d'/' -f1) 
                if [[ -n "$wifi_name" && -n "$ip_address" ]]; then
                    break
                fi
            fi
            sleep 2
            attempt=$((attempt + 1))
        done

        if [[ -n "$uuid" && -n "$wifi_name" && -n "$ip_address" ]]; then
            ip_zh_cn=$(ip_zh_cn_text "$ip_address")
            tips_text="WIFI 连接成功, 连接的 WIFI 是: \"$wifi_name\", I P 地址是: \"$ip_zh_cn\""
            echo "tips text: $tips_text"
            play_text_audio "$tips_text"
        else
            echo -e "\033[31m❌ => Failed to get Wi-Fi information after $max_attempts attempts.\033[0m"
        fi
    else
        echo -e "\033[31m❌ => The \033[33m'kuavo-wifi-announce'\033[0m is not installed. Please install it first.\033[0m"
        exit 1
    fi
}

###############
# Main Programs
###############

# Function to display available Wi-Fi networks with signal strength >= 80%
function show_wifi() {
    echo -e "\033[32m\n✅ => Available Wi-Fi networks with signal strength >= 80%:\n\033[0m"
    nmcli dev wifi | awk '$7 >= 80'
}

# Function to connect to a specified Wi-Fi network
function connect_wifi() {
    local ssid=$3
    local password=$5

    if [[ -z "$ssid" ]]; then
        echo -e "\033[31m💡 => Usage: $0 --connect-wifi wifi \"wifi-name\" password [\"wifi-password\"]\033[0m"
        return 1
    fi

     curr_connection_uuid=$(nmcli -t -f TYPE,UUID connection show --active | grep '802-11-wireless'| grep ':' | cut -d: -f2)

    if [[ -z "$password" ]]; then
        echo -e "\033[33m\n💥 Connecting to Wi-Fi: '$ssid'\n\033[0m"
        nmcli dev wifi connect "$ssid"
    else
        echo -e "\033[33m\n💥 Connecting to Wi-Fi: '$ssid', password: '$password'\n\033[0m"
        nmcli dev wifi connect "$ssid" password "$password"
    fi

    active_ssid=$(nmcli -t -f SSID,IN-USE dev wifi | grep '*' | cut -d: -f1)
    echo "Active SSID: $active_ssid"
    if [[ "$active_ssid" == "$ssid" ]]; then
        echo "Connected successfully to $ssid"
        tips_wifi_connected
    else
        echo "Failed to connect to $ssid"
        # reconnect to previous connection.
        nmcli con up "$curr_connection_uuid"
        tips_text="连接 WIFI \""$ssid\"" 失败, 请检查 WIFI 名称或密码是否正确"
        play_text_audio "$tips_text"
    fi
}


# Main program
echo -e "\033[32m\n🤖  KUAVO-WIFI-Tool \n\033[0m"
# TODO echo hello and description
echo -e "\033[32m👋 Hello! Welcome to the KUAVO-WIFI-Tool.\033[0m"

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

case $1 in
    --show-wifi | -l)
        show_wifi
        exit 0
        ;;
    --connect-wifi | -c)
        connect_wifi "$@"
        exit 0
        ;;
    --help | -h)
        usage
        exit 1
        ;;
    *)
        usage
        exit 1
        ;;
esac