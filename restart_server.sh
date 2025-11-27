#!/bin/bash
# GPU服务器重启脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "GPU服务器重启脚本"
echo "=========================================="
echo ""

# 1. 停止所有现有的uvicorn进程
echo "[1/4] 停止现有服务..."
PIDS=$(ps aux | grep "uvicorn main:app" | grep -v grep | awk '{print $2}' || true)
if [ -n "$PIDS" ]; then
    echo "找到运行中的进程: $PIDS"
    for PID in $PIDS; do
        echo "  停止进程 $PID..."
        kill "$PID" 2>/dev/null || true
    done
    sleep 2
    
    # 如果还有进程，强制杀死
    REMAINING=$(ps aux | grep "uvicorn main:app" | grep -v grep | awk '{print $2}' || true)
    if [ -n "$REMAINING" ]; then
        echo "  强制停止剩余进程..."
        for PID in $REMAINING; do
            kill -9 "$PID" 2>/dev/null || true
        done
        sleep 1
    fi
    echo "✅ 所有进程已停止"
else
    echo "✅ 没有运行中的服务"
fi

# 2. 清理PID文件
if [ -f "gpu_server.pid" ]; then
    rm -f gpu_server.pid
    echo "[2/4] ✅ 已清理PID文件"
else
    echo "[2/4] ✅ PID文件不存在"
fi

# 3. 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "❌ 错误: 虚拟环境不存在 (venv/)"
    exit 1
fi

# 4. 启动服务
echo "[3/4] 启动服务（使用新的超时配置）..."
source venv/bin/activate

# 设置环境变量
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_EXPERIMENTAL_WARNING="${HF_HUB_DISABLE_EXPERIMENTAL_WARNING:-1}"
export HF_HUB_DISABLE_VERSION_CHECK="${HF_HUB_DISABLE_VERSION_CHECK:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export FORCE_CUDA="${FORCE_CUDA:-1}"

# 创建日志目录
mkdir -p logs

# 启动服务（使用新的超时参数）
PORT="${GPU_SERVER_PORT_INTERNAL:-18001}"
nohup uvicorn main:app --host 0.0.0.0 --port "$PORT" --workers 1 \
  --timeout-keep-alive 300 --timeout-graceful-shutdown 30 \
  > logs/gpu_server.log 2>&1 &

NEW_PID=$!
echo "$NEW_PID" > gpu_server.pid

echo "✅ 服务已启动，PID: $NEW_PID"
echo "   日志文件: logs/gpu_server.log"
echo "   查看日志: tail -f logs/gpu_server.log"

# 5. 等待服务启动并验证
echo "[4/4] 验证服务..."
sleep 3

# 检查进程是否还在运行
if ps -p "$NEW_PID" > /dev/null 2>&1; then
    echo "✅ 进程运行正常"
else
    echo "❌ 警告: 进程可能已退出，请检查日志"
    tail -20 logs/gpu_server.log
    exit 1
fi

# 测试健康检查
if command -v curl >/dev/null 2>&1; then
    if curl -s -f "http://localhost:$PORT/health" > /dev/null; then
        echo "✅ 健康检查通过"
    else
        echo "⚠️  健康检查失败，但进程仍在运行"
        echo "   请稍等片刻后重试，或检查日志"
    fi
else
    echo "⚠️  curl未安装，跳过健康检查"
fi

echo ""
echo "=========================================="
echo "重启完成！"
echo "=========================================="
echo ""
echo "服务信息:"
echo "  PID: $NEW_PID"
echo "  端口: $PORT"
echo "  日志: logs/gpu_server.log"
echo ""
echo "常用命令:"
echo "  查看日志: tail -f logs/gpu_server.log"
echo "  停止服务: kill $NEW_PID"
echo "  检查状态: ps aux | grep uvicorn"

