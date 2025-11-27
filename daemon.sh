#!/bin/bash
# ================================================
# GPU Server 守护进程脚本
# 在没有systemctl的情况下保持服务运行
# ================================================

set +e  # 允许某些命令失败，以便监控循环继续

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# PID文件
PID_FILE="$SCRIPT_DIR/gpu_server.pid"
LOG_FILE="$SCRIPT_DIR/logs/daemon.log"

# 默认配置
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"
RESTART_DELAY="${RESTART_DELAY:-10}"

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 检查服务是否运行
is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0
        else
            rm -f "$PID_FILE"
            return 1
        fi
    fi
    return 1
}

# 停止服务
stop_service() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            log "停止服务 (PID: $PID)..."
            kill "$PID" 2>/dev/null || true
            sleep 2
            if ps -p "$PID" > /dev/null 2>&1; then
                log "强制停止服务..."
                kill -9 "$PID" 2>/dev/null || true
            fi
        fi
        rm -f "$PID_FILE"
    fi
}

# 启动服务
start_service() {
    log "=========================================="
    log "启动 GPU Server 守护进程"
    log "=========================================="
    
    # 检查虚拟环境
    if [ ! -d "venv" ]; then
        log "错误: 虚拟环境不存在，请先运行 setup.sh"
        return 1
    fi
    
    # 创建日志目录
    mkdir -p logs
    
    # 检查端口是否被占用
    if command -v lsof &> /dev/null; then
        if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
            OLD_PID=$(lsof -Pi :$PORT -sTCP:LISTEN -t)
            if [ "$OLD_PID" != "$(cat "$PID_FILE" 2>/dev/null)" ]; then
                log "警告: 端口 $PORT 已被进程 $OLD_PID 占用"
            fi
        fi
    fi
    
    # 激活虚拟环境
    source venv/bin/activate
    
    # 启动服务（后台运行）
    log "启动服务: HOST=$HOST PORT=$PORT WORKERS=$WORKERS"
    
    nohup uvicorn main:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level info \
        >> "$SCRIPT_DIR/logs/server.log" 2>&1 &
    
    SERVER_PID=$!
    echo "$SERVER_PID" > "$PID_FILE"
    
    log "服务已启动 (PID: $SERVER_PID)"
    
    # 等待服务启动
    sleep 3
    
    # 检查服务是否成功启动
    if ps -p "$SERVER_PID" > /dev/null 2>&1; then
        # 尝试连接服务确认启动成功
        if command -v curl &> /dev/null; then
            sleep 2
            if curl -s "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
                log "服务运行正常，健康检查通过"
            else
                log "服务进程存在，但健康检查未通过（可能仍在启动中）"
            fi
        else
            log "服务运行正常"
        fi
        return 0
    else
        log "服务启动失败"
        rm -f "$PID_FILE"
        return 1
    fi
}

# 主循环：监控并自动重启
monitor_loop() {
    log "开始监控服务..."
    
    while true; do
        if ! is_running; then
            log "检测到服务未运行，正在重启..."
            if start_service; then
                log "服务重启成功"
            else
                log "服务重启失败，等待 ${RESTART_DELAY} 秒后重试..."
                sleep "$RESTART_DELAY"
            fi
        fi
        
        # 每30秒检查一次
        sleep 30
    done
}

# 处理信号
cleanup() {
    log "收到停止信号，正在关闭服务..."
    stop_service
    exit 0
}

trap cleanup SIGTERM SIGINT

# 主函数
main() {
    case "${1:-start}" in
        start)
            if is_running; then
                PID=$(cat "$PID_FILE")
                log "服务已在运行 (PID: $PID)"
                exit 0
            fi
            
            start_service
            monitor_loop
            ;;
        stop)
            stop_service
            log "服务已停止"
            ;;
        restart)
            stop_service
            sleep 2
            start_service
            monitor_loop
            ;;
        status)
            if is_running; then
                PID=$(cat "$PID_FILE")
                log "服务正在运行 (PID: $PID)"
                exit 0
            else
                log "服务未运行"
                exit 1
            fi
            ;;
        *)
            echo "用法: $0 {start|stop|restart|status}"
            exit 1
            ;;
    esac
}

main "$@"

