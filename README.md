# GPU Model Server (PDF + Embedding + Rerank)

独立的 FastAPI 服务，用于在单独服务器上运行 `marker-pdf`、BGE Embedding、BGE Reranker 等模型，对外提供 PDF 解析、向量化和重排序能力。

## 一、功能说明

- 接口：`POST /pdf_to_markdown`
  - 请求：`multipart/form-data`，字段 `file` 为 PDF 文件
  - 响应：JSON
    - `content`: 转换后的 Markdown 文本
    - `conversion_method`: 固定为 `marker-pdf`
    - `file_name`: 原始文件名

- 接口：`POST /embed`
  - 请求：`application/json`
    - `texts`: `string[]`，待向量化的文本列表
  - 响应：JSON
    - `embeddings`: `number[][]`，与输入一一对应的向量列表

- 接口：`POST /rerank`
  - 请求：`application/json`
    - `query`: `string`，查询文本
    - `documents`: `string[]`，待重排序的文档内容列表
  - 响应：JSON
    - `scores`: `number[]`，与 `documents` 一一对应的相关性分数

- 健康检查：`GET /health`

## 二、部署步骤（在 GPU 服务器上，手动方式）

1. **准备 Python 环境**

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

2. **安装 PyTorch（根据你的 CUDA 版本选择）**

示例（CUDA 11.8）：

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu118
```

如无需 GPU，可安装 CPU 版本：

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
```

3. **安装本服务依赖**

将 `GPU_server` 目录拷贝到 GPU 服务器上，例如：`/opt/GPU_server`，然后：

```bash
cd /opt/GPU_server
pip install -r requirements.txt
```

4. **启动服务（示例：内部端口 18001）**

```bash
uvicorn main:app --host 0.0.0.0 --port 18001 --workers 1
```

- 建议先用 `workers=1`，避免重复加载大模型。
- 如需对外暴露，可通过 Nginx 做反向代理（例如对外统一暴露 16000 端口），详见下文一键脚本说明。

5. **可选：环境变量配置**

- `MARKER_USE_LLM`（默认 `false`）：是否启用 marker-pdf 的 LLM 增强。
- `PDFTEXT_WORKERS`（默认 `1`）：marker-pdf 文本提取并行 worker 数量。

在 `bash` 中示例：

```bash
export MARKER_USE_LLM=false
export PDFTEXT_WORKERS=1
export EMBED_MODEL_NAME="BAAI/bge-large-zh-v1.5"           # 默认即为此，可省略
export RERANKER_MODEL_NAME="BAAI/bge-reranker-v2-m3"      # 默认即为此，可省略
uvicorn main:app --host 0.0.0.0 --port 18001 --workers 1
```

## 三、与主后端集成方式（概要）

在主后端服务器上：

1. 在主后端服务器上，设置环境变量指向 GPU 模型服务（推荐统一使用 Nginx 暴露的 16000 端口）：

```bash
export GPU_MODEL_SERVER_URL="http://<GPU_SERVER_IP>:16000"
# 可选：分别指定不同功能的 URL（通常无需设置，统一使用 GPU_MODEL_SERVER_URL 即可）
# export GPU_PDF_SERVER_URL="http://<GPU_SERVER_IP>:16000"
# export GPU_EMBED_SERVER_URL="http://<GPU_SERVER_IP>:16000"
# export GPU_RERANK_SERVER_URL="http://<GPU_SERVER_IP>:16000"
```

2. 后端中相关逻辑会自动优先调用远程服务，失败时回退到本地模型：
   - PDF 解析：`/pdf_to_markdown`（`pdf_enhanced_processor.py`）
   - 向量化：`/embed`（`remote_embeddings.py`、`vector_service.py` 等）
   - 重排序：`/rerank`（`reranker_service.py`）

本仓库中已提供集成示例逻辑，无需额外代码改动，只需正确配置环境变量即可。

## 四、一键配置脚本（安装 + 模型下载 + Nginx 16000 端口）

在 GPU 服务器上，可以使用本目录下的 `install_gpu_server.sh` 一键完成：

- 创建 Python 虚拟环境
- 安装 GPU_server 依赖
- 预下载 embedding / rerank 模型
- 生成 Nginx 反向代理配置，**对外统一使用 16000 端口**

### 使用步骤

1. 将 `GPU_server` 目录拷贝到 GPU 服务器，例如 `/opt/GPU_server`：

```bash
cd /opt/GPU_server
chmod +x install_gpu_server.sh
sudo ./install_gpu_server.sh
```

2. 脚本执行完成后，启动应用服务（内部端口默认 18001，可通过环境变量覆盖）：

```bash
cd /opt/GPU_server
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port ${GPU_SERVER_PORT_INTERNAL:-18001} --workers 1
```

3. 通过 Nginx 对外访问 GPU 模型服务（默认）：

```text
http://<GPU_SERVER_IP>:16000
```

4. 在主后端服务器上，将 `GPU_MODEL_SERVER_URL` 指向该地址即可（见上文集成方式）。

> 提示：脚本需要 root 权限以写入 `/etc/nginx/conf.d/gpu_server.conf` 并重载 Nginx；如你的环境没有 Nginx 或有特殊限制，可参考脚本内容手动配置。

