#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REMOTE_HOST="${GPU_REMOTE_HOST:-}"
REMOTE_USER="${GPU_REMOTE_USER:-root}"
REMOTE_DIR="${GPU_REMOTE_DIR:-/opt/GPU_server}"
REMOTE_SSH_PORT="${GPU_REMOTE_SSH_PORT:-22}"

ACTION="${1:-deploy}"

if [[ -z "$REMOTE_HOST" ]]; then
  echo "GPU_REMOTE_HOST 未设置，请先导出 GPU_REMOTE_HOST 或通过 SSH 自行部署。" >&2
  exit 1
fi

rsync_code() {
  rsync -az --delete \
    --exclude 'venv/' \
    --exclude 'logs/' \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    -e "ssh -p ${REMOTE_SSH_PORT}" \
    "${ROOT_DIR}/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
}

remote_ssh() {
  ssh -p "${REMOTE_SSH_PORT}" "${REMOTE_USER}@${REMOTE_HOST}" "$@"
}

case "$ACTION" in
  deploy)
    rsync_code
    echo "代码已同步到 ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
    ;;

  start)
    MODE="${2:-systemd}"
    ASYNC_FLAG="${3:-}"
    REMOTE_CMD="cd ${REMOTE_DIR} && chmod +x start_gpu_server.sh && GPU_SERVER_PORT_INTERNAL=\"${GPU_SERVER_PORT_INTERNAL:-18001}\" HF_ENDPOINT=\"${HF_ENDPOINT:-https://hf-mirror.com}\" ./start_gpu_server.sh ${MODE}"

    if [[ "$ASYNC_FLAG" == "async" || "${GPU_REMOTE_ASYNC:-0}" == "1" ]]; then
      mkdir -p "${ROOT_DIR}/logs"
      nohup ssh -p "${REMOTE_SSH_PORT}" "${REMOTE_USER}@${REMOTE_HOST}" "bash -lc '$REMOTE_CMD'" \
        > "${ROOT_DIR}/logs/remote_start.log" 2>&1 &
      echo "已在后台触发远程启动，日志: ${ROOT_DIR}/logs/remote_start.log"
    else
      remote_ssh "bash -lc '$REMOTE_CMD'"
    fi
    ;;

  deploy_and_start)
    MODE="${2:-systemd}"
    ASYNC_FLAG="${3:-}"
    rsync_code
    "$0" start "$MODE" "$ASYNC_FLAG"
    ;;

  status)
    remote_ssh "cd ${REMOTE_DIR} && ./start_gpu_server.sh status || echo 'gpu_server 可能尚未安装或启动脚本不存在'"
    ;;

  *)
    echo "用法: $0 {deploy|start|deploy_and_start|status} [mode] [async]" >&2
    echo "  deploy           同步当前 GPU_server 代码到远程服务器" >&2
    echo "  start [mode]     在远程服务器启动 GPU 模型服务，mode 默认为 systemd" >&2
    echo "                   末尾加 async 或设置 GPU_REMOTE_ASYNC=1 可异步执行" >&2
    echo "  deploy_and_start 先 deploy 再 start" >&2
    echo "  status           查看远程 GPU 模型服务状态" >&2
    echo "环境变量:" >&2
    echo "  GPU_REMOTE_HOST      远程 GPU 服务器 IP 或域名 (必填)" >&2
    echo "  GPU_REMOTE_USER      SSH 用户名，默认 root" >&2
    echo "  GPU_REMOTE_DIR       远程部署目录，默认 /opt/GPU_server" >&2
    echo "  GPU_REMOTE_SSH_PORT  SSH 端口，默认 22" >&2
    echo "  GPU_REMOTE_ASYNC     为 1 时，start/deploy_and_start 异步执行" >&2
    exit 1
    ;;

esac
