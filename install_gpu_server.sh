#!/usr/bin/env bash
set -e

# 一键安装 GPU 模型服务脚本
# 功能：
# - 创建 Python 虚拟环境
# - 安装 PyTorch（需手动按环境选择命令）
# - 安装 GPU_server 依赖
# - 预下载 embedding / rerank 模型
# - 生成 Nginx 反向代理配置，统一对外端口 16000
#
# 使用方法（在 GPU 服务器上）：
#   cd /opt/GPU_server
#   chmod +x install_gpu_server.sh
#   sudo ./install_gpu_server.sh
#
# 可选环境变量：
#   GPU_SERVER_PORT_INTERNAL   内部 uvicorn 端口（默认 18001）
#   GPU_SERVER_PORT_PUBLIC     对外暴露端口（默认 16000）
#   GPU_SERVER_SERVER_NAME     Nginx server_name（默认 _ ）

GPU_SERVER_PORT_INTERNAL=${GPU_SERVER_PORT_INTERNAL:-18001}
GPU_SERVER_PORT_PUBLIC=${GPU_SERVER_PORT_PUBLIC:-16000}
GPU_SERVER_SERVER_NAME=${GPU_SERVER_SERVER_NAME:-_}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/venv"
NGINX_CONF_PATH="/etc/nginx/conf.d/gpu_server.conf"

if [[ $EUID -ne 0 ]]; then
  echo "[ERROR] 请使用 root 权限运行本脚本：sudo ./install_gpu_server.sh" >&2
  exit 1
fi

echo "[INFO] 工作目录: $ROOT_DIR"

echo "[STEP] 安装基础依赖 (python3-venv, nginx, curl)"
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -y
  apt-get install -y python3-venv nginx curl
else
  echo "[WARN] 未检测到 apt-get，请手动确保已安装 python3-venv、nginx、curl。"
fi

echo "[STEP] 创建 Python 虚拟环境: $VENV_DIR"
if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "[STEP] 升级 pip"
pip install --upgrade pip

echo "[STEP] 提示：请根据 GPU 服务器的 CUDA 环境自行安装合适的 PyTorch 版本"
echo "       示例 (CUDA 11.8):"
echo "         pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \\\n  --index-url https://download.pytorch.org/whl/cu118"
echo "       示例 (仅 CPU):"
echo "         pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0"
echo
read -rp "[INPUT] 如已安装 PyTorch 或暂不安装，直接回车继续；如需现在安装，请手动执行上面的 pip 命令后再回车..." _dummy

echo "[STEP] 安装 GPU_server Python 依赖"
cd "$ROOT_DIR"
pip install -r requirements.txt

echo "[STEP] 预下载 embedding / rerank / marker 模型 (首次会较慢)"
python - << 'PYCODE'
import os
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker

EMBED_MODEL = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-large-zh-v1.5")
RERANK_MODEL = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")

print(f"[PY] 预下载 embedding 模型: {EMBED_MODEL}")
SentenceTransformer(EMBED_MODEL)

print(f"[PY] 预下载 reranker 模型: {RERANK_MODEL}")
FlagReranker(RERANK_MODEL, use_fp16=False)

print("[PY] 模型预下载完成")
PYCODE

echo "[STEP] 生成 Nginx 反向代理配置: $NGINX_CONF_PATH"
cat > "$NGINX_CONF_PATH" <<EOF
# GPU 模型服务 - 统一对外端口 ${GPU_SERVER_PORT_PUBLIC}
upstream gpu_server_backend {
    server 127.0.0.1:${GPU_SERVER_PORT_INTERNAL};
    keepalive 32;
}

server {
    listen ${GPU_SERVER_PORT_PUBLIC} default_server;
    listen [::]:${GPU_SERVER_PORT_PUBLIC} default_server;
    server_name ${GPU_SERVER_SERVER_NAME};

    access_log /var/log/nginx/gpu_server_access.log;
    error_log /var/log/nginx/gpu_server_error.log;

    client_max_body_size 500M;

    location / {
        proxy_pass http://gpu_server_backend;
        proxy_http_version 1.1;

        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        proxy_read_timeout 1200s;
        proxy_connect_timeout 75s;
        proxy_send_timeout 300s;

        proxy_request_buffering off;
        proxy_buffering off;
    }
}
EOF

echo "[STEP] 检查并重载 Nginx 配置"
if command -v nginx >/dev/null 2>&1; then
  nginx -t
  if command -v systemctl >/dev/null 2>&1; then
    systemctl reload nginx || nginx -s reload
  else
    nginx -s reload || echo "[WARN] 无法自动 reload nginx，请手动执行: nginx -s reload"
  fi
else
  echo "[WARN] 未检测到 nginx 命令，请确认 Nginx 已正确安装并在 PATH 中。"
fi

echo
cat <<TIP
[完成] GPU 模型服务安装步骤已完成。

下一步：
1. 在 GPU 服务器上启动 uvicorn（建议使用 tmux/screen 守护）：

   cd "$ROOT_DIR"
   source venv/bin/activate
   uvicorn main:app --host 0.0.0.0 --port ${GPU_SERVER_PORT_INTERNAL} --workers 1

2. 在主业务服务器上，将环境变量指向 Nginx 暴露的 16000 端口，例如：

   export GPU_MODEL_SERVER_URL="http://<GPU_SERVER_IP>:${GPU_SERVER_PORT_PUBLIC}"

3. 测试：

   curl http://<GPU_SERVER_IP>:${GPU_SERVER_PORT_PUBLIC}/docs

TIP
