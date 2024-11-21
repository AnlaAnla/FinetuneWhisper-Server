#!/bin/bash

echo "123456" | sudo -S bash -c "
  # 关闭Python的输出缓冲
  export PYTHONUNBUFFERED=1

  # 初始化 conda
  source /media/martin/DATA/miniconda3/etc/profile.d/conda.sh

  # 激活 yolov8 环境
  conda activate yolov8

  # 链接正确的 cuda 版本
  export LD_LIBRARY_PATH=/media/martin/DATA/miniconda3/envs/yolov8/lib/python3.8/site-packages/nvidia/cudnn/lib

  # 运行 Python 脚本并将输出重定向到 log.txt
  python main.py -u |& tee log.txt
"