#!/bin/bash

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
NODE_NAME=$(basename "$1")
LOG_DIR="$HOME/.ros/stdout/${TIMESTAMP}"

mkdir -p "${LOG_DIR}"

# 所有节点的日志都输出到同一个文件
LOGFILE="${LOG_DIR}/stdout.log" 
# # 每个节点的日志单独输出到文件
# LOGFILE="${LOG_DIR}/${NODE_NAME}.log" 

trap 'echo "Script interrupted at $(date)" >> ${LOGFILE}' INT

exec stdbuf -o0 -e0 "$@" 2>&1 | stdbuf -i0 -o0 -e0 tee -a ${LOGFILE}
