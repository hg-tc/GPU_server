# GPU Model Server (PDF + Embedding + Rerank)

独立的 FastAPI 服务，用于在单独服务器上运行 `marker-pdf`、BGE Embedding、BGE Reranker 等模型，对外提供 PDF 解析、向量化和重排序能力。

## 📋 目录

- [功能说明](#功能说明)
- [快速开始](#快速开始)
- [详细部署](#详细部署)
- [服务管理](#服务管理)
- [开机自启配置](#开机自启配置)
- [与主后端集成](#与主后端集成)
- [环境变量配置](#环境变量配置)
- [故障排查](#故障排查)

## 🚀 功能说明

### API 接口

#### 1. PDF 转 Markdown
- **接口**: `POST /pdf_to_markdown`
- **请求**: `multipart/form-data`，字段 `file` 为 PDF 文件
- **响应**: JSON
  ```json
  {
    "content": "转换后的 Markdown 文本",
    "conversion_method": "marker-pdf",
    "file_name": "原始文件名"
  }
  ```

#### 2. 文本向量化
- **接口**: `POST /embed`
- **请求**: JSON
  ```json
  {
    "texts": ["文本1", "文本2", ...]
  }
  ```
- **响应**: JSON
  ```json
  {
    "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
  }
  ```

#### 3. 文档重排序
- **接口**: `POST /rerank`
- **请求**: JSON
  ```json
  {
    "query": "查询文本",
    "documents": ["文档1", "文档2", ...]
  }
  ```
- **响应**: JSON
  ```json
  {
    "scores": [0.95, 0.87, ...]
  }
  ```

#### 4. 健康检查
- **接口**: `GET /health`
- **响应**: `{"status": "ok"}`

## 🎯 快速开始

### 一键安装（推荐）

```bash
# 1. 将 GPU_server 目录拷贝到服务器，例如 /opt/GPU_server
cd /opt/GPU_server

# 2. 运行安装脚本（需要 root 权限）
chmod +x install_gpu_server.sh
sudo ./install_gpu_server.sh

# 3. 启动服务（自动选择最佳方式）
sudo ./start_gpu_server.sh systemd

# 4. 配置开机自启（如果没有 systemd）
sudo ./start_gpu_server.sh enable-autostart

# 5. 测试服务
curl http://localhost:16000/health
```

安装脚本会自动完成：
- ✅ 创建 Python 虚拟环境
- ✅ 安装所有依赖
- ✅ 配置 Hugging Face 镜像源（解决国内访问问题）
- ✅ 预下载模型文件
- ✅ 配置 Nginx 反向代理（端口 16000）
- ✅ 准备后台启动脚本

## 📦 详细部署

### 方式一：使用安装脚本（推荐）

```bash
cd /opt/GPU_server
chmod +x install_gpu_server.sh
sudo ./install_gpu_server.sh
```

**安装脚本功能：**
- 安装基础依赖（python3-venv, nginx, curl）
- 创建 Python 虚拟环境
- 安装项目依赖
- 配置 Hugging Face 镜像源（默认使用 `https://hf-mirror.com`）
- 预下载模型文件（首次较慢）
- 生成 Nginx 配置（对外端口 16000）
- 准备启动脚本

**可选环境变量：**
```bash
export GPU_SERVER_PORT_INTERNAL=18001    # 内部端口（默认 18001）
export GPU_SERVER_PORT_PUBLIC=16000     # 对外端口（默认 16000）
export GPU_SERVER_SERVER_NAME=_         # Nginx server_name（默认 _）
export HF_ENDPOINT=https://hf-mirror.com # Hugging Face 镜像源
```

### 方式二：手动安装

#### 1. 准备 Python 环境

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

#### 2. 安装 PyTorch

**CUDA 11.8:**
```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu118
```

**仅 CPU:**
```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
```

#### 3. 安装项目依赖

```bash
pip install -r requirements.txt
```

#### 4. 配置 Hugging Face 镜像源

```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

## 🔧 服务管理

### 启动服务

项目提供了 `start_gpu_server.sh` 脚本，支持多种启动方式：

#### 1. Systemd 服务管理（推荐，支持 systemd 的系统）

```bash
sudo ./start_gpu_server.sh systemd
```

**特性：**
- ✅ 自动重启（服务异常退出时）
- ✅ 开机自启
- ✅ 日志管理（journalctl）
- ✅ 资源限制配置

**注意：** 如果系统没有 `systemctl` 命令，脚本会自动回退到后台启动模式。

#### 2. 后台启动（nohup）

```bash
./start_gpu_server.sh background
```

**特性：**
- ✅ 适合临时运行或测试
- ✅ 适合没有 systemd 的系统（如容器环境）
- ✅ 日志保存在 `logs/gpu_server.log`

#### 3. 前台启动（调试模式）

```bash
./start_gpu_server.sh manual
```

**特性：**
- ✅ 适合调试和开发
- ✅ 直接查看输出日志

### 服务管理命令

```bash
# 查看服务状态
./start_gpu_server.sh status

# 停止服务
./start_gpu_server.sh stop

# 重启服务
./start_gpu_server.sh restart [systemd|background|manual]

# 配置开机自启（无 systemd 环境）
sudo ./start_gpu_server.sh enable-autostart

# 禁用开机自启
sudo ./start_gpu_server.sh disable-autostart
```

### Systemd 服务管理

如果使用 systemd 模式，可以使用标准命令：

```bash
# 查看服务状态
systemctl status gpu-server

# 查看实时日志
journalctl -u gpu-server -f

# 查看最近 100 行日志
journalctl -u gpu-server -n 100

# 停止/启动/重启服务
systemctl stop gpu-server
systemctl start gpu-server
systemctl restart gpu-server

# 禁用/启用开机自启
systemctl disable gpu-server
systemctl enable gpu-server
```

### Init.d 服务管理（无 systemd 环境）

配置开机自启后，可以使用标准的 init.d 命令：

```bash
# 启动服务
/etc/init.d/gpu-server start

# 停止服务
/etc/init.d/gpu-server stop

# 重启服务
/etc/init.d/gpu-server restart

# 查看状态
/etc/init.d/gpu-server status
```

## 🔄 开机自启配置

### 有 Systemd 的系统

使用 systemd 模式启动后，服务会自动配置开机自启：

```bash
sudo ./start_gpu_server.sh systemd
# systemd 会自动启用开机自启
```

### 无 Systemd 的系统

#### 方式 1：使用 enable-autostart 命令（推荐）

```bash
sudo ./start_gpu_server.sh enable-autostart
```

此命令会：
- ✅ 添加 cron `@reboot` 任务（备用保障）
- ✅ 创建 `/etc/init.d/gpu-server` 脚本
- ✅ 使用 `chkconfig` 或 `update-rc.d` 启用开机自启

**双重保障机制：**
- 主要方式：init.d 脚本在系统启动时自动执行
- 备用方式：cron @reboot 在系统完全启动后执行（延迟 30 秒）

#### 方式 2：手动配置

**使用 cron @reboot:**
```bash
sudo crontab -e
# 添加以下行：
@reboot sleep 30 && /opt/GPU_server/start_gpu_server.sh background
```

**使用 init.d 脚本:**
```bash
# 复制启动脚本到 /etc/init.d/
sudo cp /opt/GPU_server/gpu-server.service /etc/init.d/gpu-server
sudo chmod +x /etc/init.d/gpu-server

# 启用开机自启
sudo update-rc.d gpu-server defaults  # Debian/Ubuntu
# 或
sudo chkconfig gpu-server on          # CentOS/RHEL
```

### 验证开机自启配置

```bash
# 检查 cron 任务
sudo crontab -l | grep gpu

# 检查 init.d 脚本
ls -la /etc/init.d/gpu-server

# 检查运行级别链接
ls -la /etc/rc*.d/*gpu-server
```

## 🔗 与主后端集成

### 在主后端服务器上配置

1. **设置环境变量指向 GPU 模型服务：**

```bash
export GPU_MODEL_SERVER_URL="http://<GPU_SERVER_IP>:16000"
```

2. **可选：分别指定不同功能的 URL（通常无需设置）：**

```bash
# export GPU_PDF_SERVER_URL="http://<GPU_SERVER_IP>:16000"
# export GPU_EMBED_SERVER_URL="http://<GPU_SERVER_IP>:16000"
# export GPU_RERANK_SERVER_URL="http://<GPU_SERVER_IP>:16000"
```

3. **后端会自动调用远程服务：**
   - PDF 解析：`/pdf_to_markdown`
   - 向量化：`/embed`
   - 重排序：`/rerank`

### 测试连接

```bash
# 健康检查
curl http://<GPU_SERVER_IP>:16000/health

# 查看 API 文档
curl http://<GPU_SERVER_IP>:16000/docs
```

## ⚙️ 环境变量配置

### 服务配置

```bash
# 服务端口
export GPU_SERVER_PORT_INTERNAL=18001    # 内部 uvicorn 端口

# Hugging Face 镜像源（解决国内访问问题）
export HF_ENDPOINT="https://hf-mirror.com"
export HF_MIRROR_ENDPOINT="https://hf-mirror.com"  # 备用

# 模型配置
export EMBED_MODEL_NAME="BAAI/bge-large-zh-v1.5"
export RERANKER_MODEL_NAME="BAAI/bge-reranker-v2-m3"

# Marker PDF 配置
export MARKER_USE_LLM=false              # 是否启用 LLM 增强
export PDFTEXT_WORKERS=1                 # PDF 文本提取并行 worker 数量

# GPU 配置
export CUDA_VISIBLE_DEVICES=0            # 指定使用的 GPU
export FORCE_CUDA=1                      # 强制使用 CUDA（即使未检测到）
```

### 在启动脚本中使用

```bash
# 设置环境变量后启动
export HF_ENDPOINT="https://hf-mirror.com"
sudo ./start_gpu_server.sh systemd
```

### 在 Systemd 服务中配置

编辑服务文件：
```bash
sudo nano /etc/systemd/system/gpu-server.service
```

在 `[Service]` 部分添加：
```ini
Environment="HF_ENDPOINT=https://hf-mirror.com"
Environment="EMBED_MODEL_NAME=BAAI/bge-large-zh-v1.5"
```

重载并重启：
```bash
sudo systemctl daemon-reload
sudo systemctl restart gpu-server
```

## 🐛 故障排查

### 服务无法启动

1. **检查服务状态：**
```bash
./start_gpu_server.sh status
```

2. **查看日志：**
```bash
# Systemd 模式
journalctl -u gpu-server -n 50

# 后台模式
tail -f logs/gpu_server.log
```

3. **检查端口占用：**
```bash
netstat -tlnp | grep 18001
# 或
lsof -i :18001
```

### 模型下载失败

1. **检查网络连接：**
```bash
curl -I https://hf-mirror.com
```

2. **配置镜像源：**
```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

3. **手动下载模型：**
```bash
source venv/bin/activate
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-zh-v1.5')"
```

### Nginx 配置问题

1. **检查 Nginx 配置：**
```bash
sudo nginx -t
```

2. **查看 Nginx 日志：**
```bash
tail -f /var/log/nginx/gpu_server_error.log
```

3. **重启 Nginx：**
```bash
sudo systemctl restart nginx
# 或
sudo nginx -s reload
```

### 开机自启不工作

1. **检查 cron 任务：**
```bash
sudo crontab -l | grep gpu
```

2. **检查 init.d 脚本：**
```bash
ls -la /etc/init.d/gpu-server
/etc/init.d/gpu-server status
```

3. **检查运行级别链接：**
```bash
ls -la /etc/rc*.d/*gpu-server
```

4. **重新配置：**
```bash
sudo ./start_gpu_server.sh disable-autostart
sudo ./start_gpu_server.sh enable-autostart
```

### 内存不足

如果遇到内存不足问题，可以：

1. **减少 worker 数量：**
```bash
# 在启动命令中指定
uvicorn main:app --host 0.0.0.0 --port 18001 --workers 1
```

2. **调整 Systemd 内存限制：**
编辑 `/etc/systemd/system/gpu-server.service`：
```ini
MemoryLimit=16G  # 根据实际情况调整
```

3. **使用 CPU 模式（如果不需要 GPU）：**
```bash
unset CUDA_VISIBLE_DEVICES
export FORCE_CUDA=0
```

## 📝 常见问题

### Q: 如何更改服务端口？

A: 设置环境变量后启动：
```bash
export GPU_SERVER_PORT_INTERNAL=18002
sudo ./start_gpu_server.sh systemd
```

### Q: 如何查看服务日志？

A: 
- Systemd 模式：`journalctl -u gpu-server -f`
- 后台模式：`tail -f logs/gpu_server.log`

### Q: 如何更新模型？

A: 删除模型缓存后重启服务：
```bash
rm -rf ~/.cache/huggingface/hub/models--BAAI--bge-large-zh-v1.5
sudo systemctl restart gpu-server
```

### Q: 支持哪些 Python 版本？

A: 推荐 Python 3.10+，已测试 Python 3.12。

### Q: 可以在 Docker 容器中运行吗？

A: 可以，使用 `background` 模式启动，并配置相应的开机自启机制。

## 📄 许可证

本项目遵循相应的开源许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！
