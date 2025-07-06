## kuavo_assets

kuavo_assets包存储了各个版本kuavo机器人的urdf模型、mujoco-xml模型、硬件配置文件、模型处理脚本等。

### 目录简介
- `models`
      - 该目录存储了机器人使用的模型文件，包括URDF、mesh、xml模型(mujoco)等
      - 以biped_s${ROBOT_VERSION}区分不同版本的机器人的模型
         - 编译`kuavo_assets`包时，会自动根据`~/.config/lejuconfig/TotalMassV${ROBOT_VERSION}`文件中的总质量自动修改对应的机器人模型的torso质量
         - 如未指定，则使用仓库默认的总质量
      - 修改机器人总质量只需要修改`~/.config/lejuconfig/TotalMassV${ROBOT_VERSION}`文件中的值即可，重新编译`kuavo_assets`包即可生效 
- `scripts`
      - 存储了一些模型处理相关的脚本
- `config`
      - config目录存储了各个版本机器人的硬件相关的配置文件 
