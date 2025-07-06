# KUAVO-ROS-CONTROL 安装配置脚本使用文档

## 脚本功能

本脚本用于自动化配置和安装KUAVO机器人控制系统，主要包含以下功能：

✅ PIP镜像源配置  
✅ 代码仓库克隆/更新  
✅ 机器人版本设置  
✅ 重量参数配置  
✅ 驱动板类型设置  
✅ 手臂电机配置  
✅ 末端执行器配置  
✅ VR依赖安装  
✅ 项目编译  
✅ 遥控器配置  
✅ 闭源代码清理

---

## 使用步骤

### 1. 运行脚本

```bash
wget -qO /tmp/setup-kuavo-ros-control.sh https://kuavo.lejurobot.com/statics/setup-kuavo-ros-control.sh; sudo chmod +x /tmp/setup-kuavo-ros-control.sh; /tmp/setup-kuavo-ros-control.sh
```

### 2. 交互式配置流程

1. **分支选择**  
   - 输入分支名称（直接回车默认使用master分支）
   - 输入特定commit哈希（直接回车使用最新版本）

2. **机器人参数配置**  
   - 输入机器人版本号：[40/41/42/43/44/45]
   - 输入机器人重量（kg）：输入实际重量数值
   - 输入机器人驱动板类型：elmo/youda

3. **末端执行器配置**  
   
   ```bash
   请选择末端执行器类型:
   1) 灵巧手
   2) 二指夹爪
   ```
   - 输入末端执行器类型：[1/2]

4. **H12PRO遥控器配置**  
   - 根据提示输入 y/N 决定是否配置

---

## 注意事项

1. **权限要求**  
   - 需要sudo权限执行部分操作

2. **网络依赖**  
   - 需要访问Gitee代码仓库

3. **错误处理**  
   - 脚本使用 `set -e` 遇到错误立即退出
   - 关键操作前会进行环境检查
   - 错误信息会以红色[ERROR]标出

---

## 常见问题

Q: 如何重新配置机器人参数？  
A: 直接重新运行脚本，已有配置会被覆盖更新

Q: 克隆代码仓库失败怎么办？  
A: 检查网络连接，确认能访问gitee.com，或手动克隆仓库：

```bash
cd ~
git clone https://gitee.com/leju-robot/kuavo-ros-opensource.git
git clone https://gitee.com/leju-robot/kuavo_opensource.git
```

Q: 如何验证配置是否生效？  
A: 检查配置文件：

```bash
# 版本号
echo $ROBOT_VERSION

# 重量参数
cat ~/.config/lejuconfig/TotalMassV${ROBOT_VERSION}

# 驱动板类型 
cat ~/.config/lejuconfig/EcMasterType.ini
```

---

> 💡 提示：建议在配置完成后重启系统使所有环境变量生效  
> ⚠️ 注意：本脚本会修改系统级配置，请谨慎操作生产环境
