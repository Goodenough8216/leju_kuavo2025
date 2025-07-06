###
 # @Author: dongdongmingming
 # @Date: 2024-03-28 15:46:28
 # @LastEditors: Please set LastEditors
 # @LastEditTime: 2024-04-20 14:53:53
 # @FilePath: /kuavo/tools/check_tool/update_kuavo.sh
 # @Description: 更新 当前 目录程序
### 


#!/bin/bash


# 获取当前脚本所在文件夹的绝对路径
current_script_dir=$(dirname "$(realpath "$0")")


# 使用字符串操作来截取前部分路径
kuavo_dir="${current_script_dir%/tools/check_tool}"


# 切换到仓库目录
cd ${kuavo_dir}

# 强制更新到最新版本
git fetch origin
git reset --hard origin/dev

# 切换到 dev 分支
# echo "正在切换到 dev 分支..."
git checkout -f dev

git pull origin dev --allow-unrelated-histories


sudo chown -R lab:lab .

# 提示更新完成
echo "当前目录 已将仓库中的 dev 分支内容更新到最新版本。"



