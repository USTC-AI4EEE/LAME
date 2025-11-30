#!/bin/bash

# 使用传入的第一个参数作为配置文件路径（可选）
CONFIG_FILE=${1:-"configs/32.yaml"}
if [ $# -ge 1 ]; then
  shift
fi

# 激活conda环境
source ~/.bashrc
conda activate WindSR

echo "CONFIG_FILE: $CONFIG_FILE"
echo "pwd: $(pwd)"

# 运行训练命令，使用指定的配置文件和其他传入的参数
python train.py validate --config $CONFIG_FILE "$@"
