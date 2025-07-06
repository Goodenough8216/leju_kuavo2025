---
title: "Kuavo 开机播报 WIFI 工具"
---

# Tools - announce-wifi-on-boot
## 描述
本工具用于在机器人启动时检测并语音提示连接的 WIFI 信息, 若在`3`分钟内未检测到 WIFI 信息, 则会切换到 AP 热点模式, 并语音提示热点信息。

同时, 本工具还提供了`kuavo-wifi-tool.sh`工具, 用于在热点模式下切换连接到指定 WIFI。

热点信息如下:
- 名称: `$ROBOT_NAME的热点`, 其中`$ROBOT_NAME`为安装是设置的机器人名称,
- 密码: `kuavo123456`

## 安装
> **安装用时预计 5~10 分钟.**
> 
> 如 pip 安装过慢, 您可更换国内的pip源, 如:
> ```bash
> pip config set global.trusted-host pypi.mirrors.ustc.edu.cn
> pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple
> ```

以`root`权限执行以下命令:
```bash
cd ./tools/announce_wifi/ # 请替换在你的仓库中的实际路径
sudo chmod +x ./install.sh
sudo ./install.sh --robot-name "你的机器人名称"

# ----------------
# 以下是一个安装例子:
# sudo ./install.sh --robot-name "夸父"
# 以下是安装过程部分内容输出:

🤖 ROBOT_NAME: 夸父

✅ Installing dependencies...

...
🚀🚀🚀 Installation completed in 2 minutes.

🚀🚀🚀 Success! Please reboot your system to complete the installation.

playing text: 已安装成功, 欢迎使用!
...
```
工具安装成功, 会语音提示`已安装成功, 欢迎使用!`, 下次开机时工具会自动播报WiFi信息。

## WIFI 连接工具使用

**使用场景:**
- 当机器人处于热点模式时, 可以 VNC 或 SSH 连接到机器人, 然后使用本工具切换到指定 WIFI, 连接成功后会播报 WIFI 信息,
- 机器人连接着 WIFI-A 时, 期望将其网络切换到 WIFI-B 并播报连接后的 IP 地址。
```bash
# 连接到指定WIFI
sudo /opt/lejurobot/kuavo-wifi-announce/kuavo-wifi-tool.sh --connect-wifi wifi "WIFI名称" password "WIFI密码" # 需要提供密码的情况

sudo /opt/lejurobot/kuavo-wifi-announce/kuavo-wifi-tool.sh --connect-wifi wifi "WIFI名称" # 无密码的情况

# 查看附近的 WIFI (热点模式下无法查看)
sudo /opt/lejurobot/kuavo-wifi-announce/kuavo-wifi-tool.sh --show-wifi

# 帮助
/opt/lejurobot/kuavo-wifi-announce/kuavo-wifi-tool.sh -h
# 以下内容是本工具的帮助信息:

🤖  KUAVO-WIFI-Tool 

👋 Hello! Welcome to the KUAVO-WIFI-Tool.

Usage: ./kuavo-wifi-tool.sh [option]

Options:
  -l, --show-wifi      Show available Wi-Fi networks
  -c, --connect-wifi   Connect to a Wi-Fi network
                       Usage: ./kuavo-wifi-tool.sh -c wifi "wifi-name" password ["wifi-password"]
  -h, --help           Display this help message
```

## 卸载
以`root`权限执行以下命令:
```bash
cd ./tools/announce_wifi/ # 请替换在你的仓库中的实际路径
chmod +x ./uninstall.sh
sudo ./uninstall.sh # 执行此命令卸载工具

# 或者执行 `sh /opt/lejurobot/kuavo-wifi-announce/uninstall.sh` 命令卸载

# 以下是卸载过程部分内容输出:
Removed /etc/systemd/system/multi-user.target.wants/kuavo-wifi-announce.service.

👋👋👋 Uninstall Success! 'kuavo-wifi-announce' was removed, Goodbye!
```

## 其他
**查看服务状态**
```bash
systemctl status kuavo-wifi-announce.service # 查看服务状态
```