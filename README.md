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

# 2. 启动
./start.sh

# 3. 测试
python test.py --url http://localhost:8000

# 4. 查看日志
./logs.sh
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

```bash
./start.sh

# 或手动启动
source venv/bin/activate
python main.py
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

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| HOST | 0.0.0.0 | 监听地址 |
| PORT | 8000 | 监听端口 |
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
