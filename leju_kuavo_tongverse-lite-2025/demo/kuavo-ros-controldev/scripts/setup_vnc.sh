#!/bin/bash

# 安装 TightVNC Server
# sudo apt-get update
sudo apt-get install -y tigervnc-standalone-server 
# sudo apt-get install -y xfce4 xfce4-terminal xvfb

# 设置使用xfce4桌面
# echo "" > ~/.vnc/xstartup && \
# echo "#!/bin/bash" > ~/.vnc/xstartup && \
# echo "startxfce4 &" >> ~/.vnc/xstartup && \
# echo "xsetroot -solid grey" >> ~/.vnc/xstartup && \

echo '#!/bin/sh
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS

export XDG_SESSION_TYPE=x11
xsetroot -solid grey
gnome-session &
' > ~/.vnc/xstartup


chmod +x ~/.vnc/xstartup

# 
# 设置使用xvfb的
# echo "" > ~/.vnc/xstartup && \
# echo "#!/bin/bash" > ~/.vnc/xstartup && \
# echo "Xvfb :0 -screen 0 1024x768x16 &"  >> ~/.vnc/xstartup && \
# echo "export DISPLAY=:0"  >> ~/.vnc/xstartup && \
# echo "startxfce4 &" >> ~/.vnc/xstartup && \
# echo "xsetroot -solid grey" >> ~/.vnc/xstartup && \
# chmod +x ~/.vnc/xstartup
# 创建服务文件
sudo tee /etc/systemd/system/tightvncserver.service > /dev/null <<EOL
[Unit]
Description=TightVNC Server
After=network.target

[Service]
Type=forking
ExecStart=/usr/bin/vncserver :1 -geometry 1920x1080 -localhost no -depth 24
User=$USER
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOL

# 重新加载 systemd 配置
sudo systemctl daemon-reload

# 启用服务并设置开机自启动
sudo systemctl enable tightvncserver.service

# 启动服务
vncserver --kill :1
sudo vncserver --kill :1
sudo systemctl restart tightvncserver.service
