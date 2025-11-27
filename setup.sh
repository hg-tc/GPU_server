#!/bin/bash
# ================================================
# PaddleOCR 3.x GPU Server 安装脚本
# ================================================

set -e

echo "=========================================="
echo "  PaddleOCR 3.x GPU Server 安装"
echo "=========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检测 Python 版本
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}错误: 未找到 Python${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}检测到 Python 版本: $PYTHON_VERSION${NC}"

# 检测 CUDA 版本
CUDA_VERSION=""
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo -e "${GREEN}检测到 CUDA 版本: $CUDA_VERSION${NC}"
elif [ -f /usr/local/cuda/version.txt ]; then
    CUDA_VERSION=$(cat /usr/local/cuda/version.txt | awk '{print $3}')
    echo -e "${GREEN}检测到 CUDA 版本: $CUDA_VERSION${NC}"
else
    echo -e "${YELLOW}警告: 未检测到 CUDA，将安装 CPU 版本${NC}"
fi

# 创建虚拟环境
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${GREEN}创建虚拟环境...${NC}"
    $PYTHON_CMD -m venv $VENV_DIR
fi

# 激活虚拟环境
source $VENV_DIR/bin/activate

# 升级 pip
echo -e "${GREEN}升级 pip...${NC}"
pip install --upgrade pip

# 安装 PaddlePaddle
echo -e "${GREEN}安装 PaddlePaddle...${NC}"
if [ -n "$CUDA_VERSION" ]; then
    # 根据 CUDA 版本选择安装源
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
    
    if [ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 8 ]; then
        echo -e "${GREEN}安装 PaddlePaddle GPU (CUDA 11.8)...${NC}"
        pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
    elif [ "$CUDA_MAJOR" -eq 12 ]; then
        echo -e "${GREEN}安装 PaddlePaddle GPU (CUDA 12.x)...${NC}"
        pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
    else
        echo -e "${YELLOW}CUDA 版本 $CUDA_VERSION 可能不兼容，尝试安装 CUDA 11.8 版本...${NC}"
        pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
    fi
else
    echo -e "${GREEN}安装 PaddlePaddle CPU 版本...${NC}"
    pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
fi

# 安装 PaddleOCR
echo -e "${GREEN}安装 PaddleOCR 3.x...${NC}"
pip install "paddleocr[all]"

# 安装其他依赖
echo -e "${GREEN}安装其他依赖...${NC}"
pip install fastapi>=0.115.0
pip install "uvicorn[standard]>=0.24.0"
pip install python-multipart>=0.0.6
pip install "pydantic>=2.4.2,<3.0.0"
pip install "pydantic-settings>=2.0.3,<3.0.0"
pip install "sentence-transformers>=3.0.0"
pip install "FlagEmbedding>=1.3.5"
pip install "numpy>=1.24.0,<2.0.0"
pip install "Pillow>=10.0.0"
pip install "opencv-python>=4.8.0"

# 验证安装
echo ""
echo -e "${GREEN}验证安装...${NC}"
echo "=========================================="

# 验证 PaddlePaddle
$PYTHON_CMD -c "import paddle; print(f'PaddlePaddle 版本: {paddle.__version__}')"
$PYTHON_CMD -c "import paddle; print(f'GPU 可用: {paddle.device.is_compiled_with_cuda()}')"

# 验证 PaddleOCR
$PYTHON_CMD -c "from paddleocr import PaddleOCR; print('PaddleOCR 导入成功')"

# 验证 PP-StructureV3
$PYTHON_CMD -c "from paddleocr import PPStructureV3; print('PPStructureV3 导入成功')"

# ================================================
# 预下载模型
# ================================================
echo ""
echo -e "${GREEN}=========================================="
echo "  预下载模型"
echo "==========================================${NC}"

# 设置 HuggingFace 镜像（可选，国内加速）
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
echo "HuggingFace 镜像: $HF_ENDPOINT"

# 下载 PaddleOCR 模型
echo ""
echo -e "${GREEN}下载 PaddleOCR 模型...${NC}"
$PYTHON_CMD << 'EOF'
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

print("初始化 PaddleOCR (PP-OCRv5)...")
from paddleocr import PaddleOCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
print("✅ PaddleOCR 模型下载完成")
EOF

# 下载 PP-StructureV3 模型
echo ""
echo -e "${GREEN}下载 PP-StructureV3 模型...${NC}"
$PYTHON_CMD << 'EOF'
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

print("初始化 PP-StructureV3...")
from paddleocr import PPStructureV3
structure = PPStructureV3()
print("✅ PP-StructureV3 模型下载完成")
EOF

# 下载 Embedding 模型
echo ""
echo -e "${GREEN}下载 Embedding 模型 (BGE-large-zh-v1.5)...${NC}"
$PYTHON_CMD << 'EOF'
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

print("下载 BAAI/bge-large-zh-v1.5...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
# 测试一下
embeddings = model.encode(["测试文本"])
print(f"✅ Embedding 模型下载完成，向量维度: {len(embeddings[0])}")
EOF

# 下载 Reranker 模型
echo ""
echo -e "${GREEN}下载 Reranker 模型 (BGE-reranker-v2-m3)...${NC}"
$PYTHON_CMD << 'EOF'
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

print("下载 BAAI/bge-reranker-v2-m3...")
from FlagEmbedding import FlagReranker
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
# 测试一下
scores = reranker.compute_score([["查询", "文档"]])
print(f"✅ Reranker 模型下载完成")
EOF

echo ""
echo -e "${GREEN}=========================================="
echo "  安装完成！所有模型已下载"
echo "==========================================${NC}"
echo ""
echo "启动服务器:"
echo "  ./start.sh"
echo ""
echo "或手动启动:"
echo "  source venv/bin/activate"
echo "  python main.py"
