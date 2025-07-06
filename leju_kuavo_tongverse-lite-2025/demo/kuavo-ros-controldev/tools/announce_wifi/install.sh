#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
INSTALL_DIR=/opt/lejurobot
APP_NAME=kuavo-wifi-announce
TTS_MODEL_OSS_URL="https://kuavo.lejurobot.com/models/tts_model-speech_sambert-hifigan_tts_zhida_zh-cn_16k.tar.gz"
TTS_MODEL_MD5_OSS_URL="https://kuavo.lejurobot.com/models/tts_model-speech_sambert-hifigan_tts_zhida_zh-cn_16k.tar.gz.md5"

start_time=$(date +%s)

function echo_info() {
  local message=$1
  echo -e "$message"
}

function echo_warn() {
  local message=$1
  echo -e "\033[33m$message\033[0m"
}

function echo_error() {
  local message=$1
  echo -e "\033[31m$message\033[0m"
}

function echo_success() {
  local message=$1
  echo -e "\033[32m$message\033[0m"
}

function check_root() {
  if [ $(id -u) != "0" ]; then
    echo_error "You must be root to run this script, please use root to install."
    exit 1
  fi
}

function usage() {
  echo "Usage: $0 -robot_name \"your_robot_name\""
  echo "Options:"
  echo "  -robot-name <name>  Set the name of the robot (required)."
  echo "  -h                  Show this help message and exit."
}

################### main ##################

check_root
robot_name=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --robot-name|-r)
        if [[ -n $2 && $2 != -* ]]; then
            robot_name="$2"
            shift
            shift
        else
            echo "Error: -robot_name requires a non-empty value."
            usage
            exit 1
        fi
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Error: Unknown option: $1"
        usage
        exit 1
        ;;
  esac
done

if [[ "$robot_name" == "" ]]; then
  echo_error "Error: -robot_name is required. Please specify the robot name."
  usage
  exit 1
fi

echo_warn "\n🤖 ROBOT_NAME: $robot_name\n"

# @@@ DEPENDENCIES!
echo_warn "✅ Installing dependencies..."
sudo apt-get update || { echo_error "apt-get update failed, exiting..."; exit 1; }
sudo apt-get install -y network-manager iw portaudio19-dev|| { echo_error "apt-get install failed, exiting..."; exit 1; }

# @@@ PYTHON ENV!
if ! command -v python3 &> /dev/null; then
    echo_error "Python3 is not installed. Please install Python3 and try again."
    exit 1
fi

python_version=$(python3 --version | cut -d ' ' -f 2 | cut -d '.' -f 1-2)
sudo apt-get install -y python${python_version}-venv || { echo_error "ERROR: apt-get install python${python_version}-venv failed, exiting..."; exit 1; }

VENV_PATH="$INSTALL_DIR/$APP_NAME/venv"
if [ -f "$VENV_PATH/bin/activate" ]; then
    echo_info "Python virtual environment already exists at $VENV_PATH"
else
    rm -rf "$VENV_PATH"
    echo_warn "✅ Creating Python3 venv at $VENV_PATH..."
    python3 -m venv "$VENV_PATH"
    if [ $? -ne 0 ]; then
        echo_error "Failed to create Python virtual environment at $VENV_PATH"
        exit 1
    fi
fi

source $VENV_PATH/bin/activate
# TODO check pip install success！
pip install networkx==2.8.8 # fix ERROR: Package 'networkx' requires a different Python: 3.8.10 not in '>=3.9'
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install modelscope
pip install -r requirements.txt
pip install kantts -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html 

#@@@ COPY-FILES!
echo_warn "✅ Copying files..."
if [ ! -d "$INSTALL_DIR" ]; then
  mkdir -p "$INSTALL_DIR"
fi
mkdir -p "$INSTALL_DIR/$APP_NAME/"
mkdir -p "$INSTALL_DIR/$APP_NAME/bin/"
mkdir -p "$INSTALL_DIR/$APP_NAME/config/"
mkdir -p "$INSTALL_DIR/$APP_NAME/data/model/"

# download tts-model 
echo_warn "✅ Downloading TTS model..."
tts_model_file="/tmp/tts_model-speech_sambert-hifigan_tts_zhida_zh-cn_16k.tar.gz"
if [ -f "$tts_model_file" ]; then
    echo_warn "TTS model file already exists, skipping download."
else
    wget $TTS_MODEL_OSS_URL -O $tts_model_file
    if [ $? -ne 0 ]; then
        echo_error "Failed to download the TTS model!"
        exit 1
    fi
fi

# md5 check
echo_info "✅ Checking MD5 of the TTS model..."
wget -q -O - $TTS_MODEL_MD5_OSS_URL > /tmp/tts_model.md5
if [ $? -ne 0 ]; then
    echo_error "Failed to download the MD5 file from $TTS_MODEL_MD5_OSS_URL"
    exit 1
fi
expected_md5=$(cat /tmp/tts_model.md5)
if [ -z "$expected_md5" ]; then
    echo_error "Expected MD5 is empty!"
    exit 1
fi
actual_md5=$(md5sum $tts_model_file | awk '{print $1}')
if [ "$actual_md5" != "$expected_md5" ]; then
    echo_error "MD5 checksum verification failed for the TTS model, Please try again."
    rm -rf $tts_model_file
    exit 1
fi
echo_success "🎉 MD5 checksum verification passed for the TTS model!"

# extract tts model
tar -xvf $tts_model_file -C "$INSTALL_DIR/$APP_NAME/data/model/"
if [ $? -ne 0 ]; then
    echo_error "Failed to extract the TTS model file"
    rm -f $tts_model_file
    exit 1
fi

# copy scripts
cp -f "$SCRIPT_DIR/uninstall.sh" "$INSTALL_DIR/$APP_NAME/uninstall.sh"
chmod +x "$INSTALL_DIR/$APP_NAME/uninstall.sh"
cp -f "$SCRIPT_DIR/kuavo-wifi-tool.sh" "$INSTALL_DIR/$APP_NAME/kuavo-wifi-tool.sh"
chmod +x "$INSTALL_DIR/$APP_NAME/kuavo-wifi-tool.sh"

cp -rf $SCRIPT_DIR/config/* $INSTALL_DIR/$APP_NAME/config/
cp -rf $SCRIPT_DIR/scripts/* $INSTALL_DIR/$APP_NAME/bin/
chmod +x $INSTALL_DIR/$APP_NAME/bin/start.sh
chmod +x $INSTALL_DIR/$APP_NAME/bin/test.sh
cp -f $SCRIPT_DIR/services/kuavo-wifi-announce.service /etc/systemd/system/

# @@@ CONFIG!
sed -i "s/\placeholder@robot_name/$robot_name/g" "$INSTALL_DIR/$APP_NAME/config/config.json"

#@@@ WIFI-AP!
wlan0=$(iw dev | awk '$1=="Interface" && $2 !~ /^(ap|lo|docker|veth)/{print $2; exit}')
supported_ap_mode=$(iw list | grep -A 20 'Supported interface modes' | grep '* AP')
connection_exists() {
    nmcli connection show | grep -q "$1"
}
if [ -n "$supported_ap_mode" ] && [ -n "$wlan0" ]; then
  echo_warn "Your wireless card supports AP mode with interface '$wlan0'"
  CON_NAME="kuavo-hotspot"
  AP_SSID="$robot_name的热点"
  AP_PASSWORD="kuavo123456"
  
  if connection_exists "kuavo-hotspot"; then
    echo_warn "AP(Hotspot) :'$CON_NAME' exists, updating, set SSID:'$AP_SSID', Password:'$AP_PASSWORD'"
  else
    echo_warn "✅ Creating AP(Hotspot) conection: '$CON_NAME', SSID: '$AP_SSID', Password: '$AP_PASSWORD'"
    nmcli con add type wifi ifname $wlan0  con-name $CON_NAME ssid $AP_SSID autoconnect no
  fi
  nmcli con modify $CON_NAME 802-11-wireless.mode ap 802-11-wireless.band bg ipv4.method shared
  nmcli con modify $CON_NAME wifi-sec.key-mgmt wpa-psk wifi-sec.psk $AP_PASSWORD
  nmcli con modify "$CON_NAME" 802-11-wireless.ssid "$AP_SSID"
else
  echo_warn "Your wireless card does not support AP mode."
fi

#@@@ SERVICES!
echo_warn "✅ Adding Service..."
systemctl daemon-reload
systemctl enable kuavo-wifi-announce.service

#@@@ END! 
end_time=$(date +%s)
elapsed_time=$(( (end_time - start_time) / 60 )) # Convert to minutes
elapsed_time="${elapsed_time%.*}"

echo_success "\n🚀🚀🚀 Installation completed in $elapsed_time minutes."
echo_success "\n🚀🚀🚀 Success! Please reboot your system to complete the installation.\n"

####################
# @TEST
####################
$INSTALL_DIR/$APP_NAME/bin/test.sh "已安装成功, 欢迎使用!"