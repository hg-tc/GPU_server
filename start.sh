#!/bin/bash
# ================================================
# PaddleOCR 3.x GPU Server 启动脚本
# ================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 默认配置
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo -e "${RED}错误: 虚拟环境不存在，请先运行 setup.sh${NC}"
    exit 1
fi

# 激活虚拟环境
source venv/bin/activate

# 创建日志目录
mkdir -p logs

echo -e "${GREEN}=========================================="
echo "  PaddleOCR 3.x GPU Server"
echo "==========================================${NC}"
echo ""
echo "配置:"
echo "  - Host: $HOST"
echo "  - Port: $PORT"
echo "  - Workers: $WORKERS"
echo ""

# 检查 GPU
python3 -c "
import paddle
if paddle.device.is_compiled_with_cuda():
    print('GPU: 可用 (PaddlePaddle CUDA)')
else:
    print('GPU: 不可用，使用 CPU')
" 2>/dev/null || echo "GPU: 检测失败"

echo ""
echo -e "${GREEN}启动服务器...${NC}"
echo ""

# 启动服务器
exec uvicorn main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level info
