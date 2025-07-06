#!/bin/bash
# 运行dstat，收集系统资源使用情况，并将结果输出到CSV文件
current_date=$(date +"%Y-%m-%d_%H-%M-%S")
dstat -cdnmgy --output /tmp/dstat_output_$current_date.csv 1 > /dev/null
