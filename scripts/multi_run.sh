#########################################################################
# File Name: scripts/multi_run.sh
# Author: frank
# mail: 1216451203@qq.com
# Created Time: 2025年07月22日 星期二 21时18分09秒
#########################################################################
#!/bin/bash

# 设置默认并发数（允许同时运行的最大进程数）
CONCURRENCY=3

# 设置要运行的总任务数
TOTAL_TASKS=10000

# 使用函数显示帮助信息
usage() {
  echo "用法: $0 [-c 并发数] [-t 总任务数]"
  echo "示例: $0 -c 10 -t 100"
  exit 1
}

# 解析命令行参数
while getopts ":c:t:" opt; do
  case $opt in
  c) CONCURRENCY="$OPTARG" ;;
  t) TOTAL_TASKS="$OPTARG" ;;
  \?) usage ;;
  esac
done

# 验证参数有效性
if ! [[ "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || ! [[ "$TOTAL_TASKS" =~ ^[0-9]+$ ]]; then
  echo "错误：参数必须是正整数"
  usage
fi

echo "启动并发执行..."
echo "并发数: $CONCURRENCY"
echo "总任务数: $TOTAL_TASKS"

# 创建命名管道用于控制并发
fifo_path="/tmp/$$.fifo"
mkfifo "$fifo_path"
exec 3<>"$fifo_path"
rm -f "$fifo_path"

# 向管道注入并发数令牌
for ((i = 0; i < CONCURRENCY; i++)); do
  echo >&3
done

# 任务计数器
count=0

# 启动所有任务
for ((i = 1; i <= TOTAL_TASKS; i++)); do
  # 读取令牌（阻塞直到有可用令牌）
  read -u 3 -t 1

  # 启动后台任务
  (
    # 执行实际任务（替换为你的命令）
    # python client.py --task_id=$i
    #random_file="LJ001-$(printf '%04d' $((RANDOM % 10 + 1))).wav"
    #random_file="$(printf 'audios/%06d' $((RANDOM % 50 + 1))).wav"
    random_file="$(printf 'audios/%06d' $(($i % 50 + 1))).wav"
    python3 offline-websocket-client-decode-files-sequential.py --server-addr localhost --server-port 6001 "$random_file"

    # 任务完成后归还令牌
    echo >&3
  ) &

  # 更新计数器并显示进度
  ((count++))
  printf "已启动任务: %d/%d (PID: %d)\r" $count $TOTAL_TASKS $!
done

# 等待所有后台任务完成
wait
echo -e "\n所有任务已完成"
exec 3>&-
