#!/bin/bash

# 定义基础算法和实验场景
BASE_ALGO="E-GAT"
EXPERIMENT_SCENE="MMM2"

# 循环创建tmux会话并执行命令
for i in {1..3}
do
    # 计算随机种子
    SEED=$((11111 * i))

    # 创建tmux会话
    SESSION_NAME="${EXPERIMENT_SCENE}-${BASE_ALGO}-${i}"
    tmux new-session -d -s $SESSION_NAME

    # 激活conda环境并执行脚本
    tmux send-keys -t $SESSION_NAME "conda activate harl" C-m
    tmux send-keys -t $SESSION_NAME "python train.py --algo mappo-HOANet --env smac --exp_name MMM_model --seed $SEED" C-m

    echo "创建了第${i}个tmux会话。"
    # 等待30秒,防止进程冲突占用
    if [ $i -ne 3 ]; then
        sleep 30s
    fi
done

echo "脚本执行完毕。创建了3个tmux会话。"
