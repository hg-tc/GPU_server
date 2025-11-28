# GPU Model Server - PaddleOCR 3.x

基于 PaddleOCR 3.x 的 GPU 模型服务器，支持 PDF 转 Markdown、图片 OCR、文档版面分析、文本嵌入和重排序。

## 功能特性

- **PDF 转 Markdown**: 使用 PP-StructureV3
- **图片 OCR**: 使用 PaddleOCR 3.x (PP-OCRv5)
- **文档版面分析**: 使用 PP-StructureV3，支持表格、公式、图表识别
- **文本嵌入**: 使用 sentence-transformers (BGE 系列)
- **文本重排序**: 使用 FlagEmbedding (BGE-Reranker)

## 快速开始

```bash
# 1. 安装
./setup.sh

# 2. 配置 Nginx 反向代理（可选，推荐）
./install_nginx.sh

# 3. 启动服务（使用守护进程，支持自动重启）
./daemon.sh start

# 4. 测试服务
# 内部端口
python test.py --url http://localhost:8000
# 或通过 Nginx 代理
python test.py --url http://localhost:16000

# 5. 查看日志
./logs.sh
# 或
tail -f logs/daemon.log
```

### 完整部署流程

```bash
# 1. 安装依赖
./setup.sh

# 2. 配置 Nginx 反向代理
./install_nginx.sh

# 3. 启动服务（使用 screen 保持运行）
screen -S gpu_server
./daemon.sh start
# 按 Ctrl+A 然后按 D 退出 screen

# 4. 验证服务
curl http://localhost:16000/health
./daemon.sh status
```

## 安装

### 自动安装

```bash
chmod +x setup.sh
./setup.sh
```

脚本会自动:
- 检测 CUDA 版本
- 创建虚拟环境
- 安装 PaddlePaddle (GPU/CPU)
- 安装 PaddleOCR 3.x
- 安装其他依赖

### 手动安装

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装 PaddlePaddle GPU (CUDA 11.8)
pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# 或 CPU 版本
# pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# 安装 PaddleOCR
pip install "paddleocr[all]"

# 安装其他依赖
pip install -r requirements.txt
```

## 启动服务

### 方式一：直接启动（前台运行）

```bash
./start.sh

# 或手动启动
source venv/bin/activate
python main.py
```

### 方式二：使用守护进程（推荐，支持自动重启）

```bash
# 启动服务（后台运行，自动监控和重启）
./daemon.sh start

# 查看服务状态
./daemon.sh status

# 停止服务
./daemon.sh stop

# 重启服务
./daemon.sh restart
# 或使用重启脚本
./restart.sh
```

### 方式三：使用 Screen 保持运行

```bash
# 创建 screen 会话
screen -S gpu_server

# 在 screen 中启动守护进程
./daemon.sh start

# 退出 screen（按 Ctrl+A 然后按 D），服务会继续运行

# 重新连接到 screen
screen -r gpu_server
```

### 方式四：使用 Tmux 保持运行

```bash
# 创建 tmux 会话
tmux new -s gpu_server

# 在 tmux 中启动守护进程
./daemon.sh start

# 退出 tmux（按 Ctrl+B 然后按 D），服务会继续运行

# 重新连接到 tmux
tmux attach -t gpu_server
```

## API 端点

### 健康检查

```bash
GET /health
```

### 图片 OCR

```bash
POST /ocr_image
Content-Type: multipart/form-data
file: <image_file>
```

响应:
```json
{
  "text": "识别的文本",
  "confidence": 0.95,
  "lines": ["行1", "行2"],
  "confidences": [0.98, 0.92],
  "boxes": [[[x1,y1], [x2,y2], ...], ...]
}
```

### Base64 图片 OCR

```bash
POST /ocr_base64
Content-Type: application/json
{
  "image_base64": "base64编码的图片",
  "filename": "image.png"
}
```

### 图片版面分析 (PP-StructureV3)

```bash
POST /structure_image
Content-Type: multipart/form-data
file: <image_file>
```

响应:
```json
{
  "markdown": "转换后的Markdown",
  "layout_info": {...}
}
```

### PDF 版面分析 (PP-StructureV3)

```bash
POST /structure_pdf
Content-Type: multipart/form-data
file: <pdf_file>
```

### PDF 转 Markdown

```bash
POST /pdf_to_markdown
Content-Type: multipart/form-data
file: <pdf_file>
```

### 文本嵌入

```bash
POST /embed
Content-Type: application/json
{
  "texts": ["文本1", "文本2"]
}
```

### 批量文本嵌入

```bash
POST /embed_batch
Content-Type: application/json
{
  "batches": [["文本1", "文本2"], ["文本3"]]
}
```

### 文档重排序

```bash
POST /rerank
Content-Type: application/json
{
  "query": "查询文本",
  "documents": ["文档1", "文档2"]
}
```

## Nginx 反向代理配置

### 安装和配置 Nginx

```bash
# 自动安装和配置
./install_nginx.sh
```

脚本会自动：
- 检测并安装 Nginx（如果未安装）
- 配置反向代理（16000 端口 → 8000 端口）
- 测试配置并启动 Nginx

### 端口说明

- **外部访问端口**: 16000（通过 Nginx 代理）
- **内部服务端口**: 8000（FastAPI 服务实际运行端口）

### 验证配置

```bash
# 测试内部服务
curl http://127.0.0.1:8000/health

# 测试外部访问（通过 Nginx）
curl http://127.0.0.1:16000/health
curl http://your-server-ip:16000/health
```

### 手动配置 Nginx

如果自动安装失败，可以手动配置：

```bash
# 复制配置文件
sudo cp nginx.conf /etc/nginx/conf.d/gpu_server.conf

# 测试配置
sudo nginx -t

# 启动/重载 Nginx
sudo systemctl start nginx
# 或
sudo nginx -s reload
```

详细配置说明请参考 [README_NGINX.md](README_NGINX.md)

## 服务管理

### 启动服务

```bash
# 使用守护进程（推荐）
./daemon.sh start

# 或使用重启脚本（会自动停止旧服务）
./restart.sh
```

### 停止服务

```bash
./daemon.sh stop
```

### 重启服务

```bash
# 方式一：使用重启脚本（推荐）
./restart.sh

# 方式二：使用 daemon.sh
./daemon.sh restart

# 方式三：手动停止后启动
./daemon.sh stop
./daemon.sh start
```

### 查看服务状态

```bash
./daemon.sh status

# 检查端口监听
netstat -tlnp | grep 8000
# 或
ss -tlnp | grep 8000
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| HOST | 0.0.0.0 | 监听地址 |
| PORT | 8000 | 监听端口 |
| WORKERS | 1 | 工作进程数 |
| RESTART_DELAY | 10 | 重启延迟（秒） |
| EMBED_MODEL_NAME | BAAI/bge-large-zh-v1.5 | 嵌入模型 |
| RERANKER_MODEL_NAME | BAAI/bge-reranker-v2-m3 | 重排序模型 |

## 本地客户端调用

在主应用中使用 `GPUClient`:

```python
from app.services.gpu_client import GPUClient

client = GPUClient.get_instance()

# 图片 OCR
result = await client.ocr_image("/path/to/image.png")
print(result["text"])

# Base64 OCR
result = await client.ocr_base64(base64_string, "image.png")

# 版面分析
result = await client.structure_image("/path/to/image.png")
print(result["markdown"])

# PDF 版面分析
result = await client.structure_pdf("/path/to/document.pdf")
```

## 日志查看

### 使用 logs.sh 脚本

```bash
# 实时查看日志
./logs.sh f

# 查看最后 100 行
./logs.sh l 100

# 搜索日志
./logs.sh g "error"

# 查看统计
./logs.sh s
```

### 直接查看日志文件

```bash
# 查看守护进程日志
tail -f logs/daemon.log

# 查看服务日志
tail -f logs/server.log

# 查看错误日志
tail -f logs/gpu_server_error.log

# 查看所有日志
tail -f logs/*.log
```

## 测试

```bash
# 运行所有测试
python test.py

# 指定服务器地址
python test.py --url http://gpu-server:8000

# 测试图片处理
python test.py --image /path/to/image.png

# 测试 PDF 处理
python test.py --pdf /path/to/document.pdf
```

## 故障排除

### CUDA 不可用

```bash
# 检查 PaddlePaddle CUDA 支持
python -c "import paddle; print(paddle.device.is_compiled_with_cuda())"

# 检查 PyTorch CUDA 支持
python -c "import torch; print(torch.cuda.is_available())"
```

### 模型下载慢

设置镜像源:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 内存不足

减少并发数或使用更小的模型:
```python
ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
)
```

### 服务无法启动

```bash
# 检查端口是否被占用
lsof -i :8000
# 或
netstat -tlnp | grep 8000

# 检查服务状态
./daemon.sh status

# 查看错误日志
tail -f logs/daemon.log
tail -f logs/gpu_server_error.log
```

### Nginx 无法访问

```bash
# 检查 Nginx 是否运行
ps aux | grep nginx
# 或
systemctl status nginx

# 检查 Nginx 配置
sudo nginx -t

# 查看 Nginx 错误日志
sudo tail -f /var/log/nginx/error.log

# 检查防火墙
sudo ufw status
# 或
sudo iptables -L
```

### 服务频繁重启

```bash
# 查看服务日志找出原因
tail -f logs/server.log
tail -f logs/daemon.log

# 检查系统资源
free -h
df -h
top

# 检查 GPU 状态
nvidia-smi
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `setup.sh` | 自动安装脚本 |
| `start.sh` | 前台启动脚本 |
| `daemon.sh` | 守护进程脚本（支持自动重启） |
| `restart.sh` | 服务重启脚本 |
| `install_nginx.sh` | Nginx 安装和配置脚本 |
| `nginx.conf` | Nginx 反向代理配置文件 |
| `main.py` | FastAPI 主程序 |
| `logs.sh` | 日志查看脚本 |
| `README_NGINX.md` | Nginx 配置详细说明 |
