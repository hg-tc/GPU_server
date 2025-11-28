#!/bin/bash
# ================================================
# GPU Server 重启脚本
# ================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PID_FILE="$SCRIPT_DIR/gpu_server.pid"

echo -e "${GREEN}=========================================="
echo "  重启 GPU Server"
echo "==========================================${NC}"

# 停止服务
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${YELLOW}停止服务 (PID: $PID)...${NC}"
        kill "$PID" 2>/dev/null || true
        sleep 2
        if ps -p "$PID" > /dev/null 2>&1; then
            echo -e "${YELLOW}强制停止服务...${NC}"
            kill -9 "$PID" 2>/dev/null || true
        fi
        echo -e "${GREEN}服务已停止${NC}"
    fi
    rm -f "$PID_FILE"
fi

# 等待端口释放
if command -v lsof &> /dev/null; then
    PORT="${PORT:-8000}"
    for i in {1..10}; do
        if ! lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
            break
        fi
        echo -e "${YELLOW}等待端口 $PORT 释放... ($i/10)${NC}"
        sleep 1
    done
fi

# 启动服务
echo -e "${GREEN}启动服务...${NC}"
./daemon.sh start

echo ""
echo -e "${GREEN}=========================================="
echo "  服务重启完成"
echo "==========================================${NC}"
echo ""
echo "查看服务状态:"
echo "  ./daemon.sh status"
echo ""
echo "查看日志:"
echo "  tail -f logs/daemon.log"
echo "  tail -f logs/server.log"
echo ""


