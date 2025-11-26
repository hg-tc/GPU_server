import logging
import os
import tempfile
import time
from datetime import datetime
from typing import List
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# 配置 Hugging Face 离线模式（如果模型已完全下载，可以避免网络请求）
# 注意：启用后如果模型文件不完整可能会失败
HF_HUB_OFFLINE = os.getenv("HF_HUB_OFFLINE", "0").lower() in {"1", "true", "yes"}
if HF_HUB_OFFLINE:
    # 完全禁用 Hugging Face Hub 的网络请求
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    # 禁用重试机制
    os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
    # 禁用版本检查
    os.environ["HF_HUB_DISABLE_VERSION_CHECK"] = "1"
else:
    # 配置 Hugging Face 镜像源（如果未设置）
    if not os.getenv("HF_ENDPOINT"):
        # 使用 hf-mirror.com 作为默认镜像源
        os.environ["HF_ENDPOINT"] = os.getenv("HF_MIRROR_ENDPOINT", "https://hf-mirror.com")

# 配置日志系统
def setup_logging():
    """配置详细的日志系统"""
    # 创建日志目录
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 日志格式
    log_format = "%(asctime)s [%(levelname)-8s] [%(name)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # 文件处理器 - 详细日志
    file_handler = logging.FileHandler(
        log_dir / "gpu_server.log",
        encoding="utf-8",
        mode="a"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # 文件处理器 - 错误日志
    error_handler = logging.FileHandler(
        log_dir / "gpu_server_error.log",
        encoding="utf-8",
        mode="a"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)
    
    # 配置应用日志记录器
    logger = logging.getLogger("gpu_pdf_server")
    logger.setLevel(logging.DEBUG)
    
    # 降低第三方库的日志级别（减少日志噪音）
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)  # 减少 Hugging Face 网络请求的 DEBUG 日志
    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)  # 减少 transformers 库的详细日志
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)  # 减少 Hugging Face Hub 的详细日志
    
    # 如果启用离线模式，完全禁用网络相关日志
    if os.getenv("HF_HUB_OFFLINE") == "1":
        logging.getLogger("urllib3").setLevel(logging.ERROR)
        logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    
    return logger

logger = setup_logging()

# 记录离线模式状态
if os.getenv("HF_HUB_OFFLINE", "0") == "1":
    logger.info("=" * 60)
    logger.info("📴 Hugging Face 离线模式已启用")
    logger.info("   模型将从本地缓存加载，不会进行网络请求")
    logger.info("=" * 60)

# 启动时检测并记录GPU信息
try:
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info("=" * 60)
        logger.info("🚀 GPU 检测信息:")
        logger.info(f"  ✅ CUDA可用: 是")
        logger.info(f"  📦 CUDA版本: {torch.version.cuda}")
        logger.info(f"  🔧 PyTorch版本: {torch.__version__}")
        logger.info(f"  🎮 GPU数量: {gpu_count}")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  🎯 GPU {i}: {props.name}")
            logger.info(f"     显存: {props.total_memory / 1024**3:.2f} GB")
            logger.info(f"     计算能力: {props.major}.{props.minor}")
        logger.info("=" * 60)
    else:
        logger.warning("=" * 60)
        logger.warning("⚠️  CUDA不可用，模型将使用CPU运行（性能较慢）")
        logger.warning("=" * 60)
except Exception as e:
    logger.warning(f"GPU检测失败: {e}")

app = FastAPI(
    title="GPU Model Server",
    version="0.1.0",
    description="Offload marker-pdf PDF->Markdown, embeddings, and reranking to a dedicated server",
)

# 请求日志中间件
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"
        
        # 记录请求开始
        logger.info(f"[请求开始] {request.method} {request.url.path} | 客户端: {client_ip}")
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # 记录请求完成
            status_code = response.status_code
            status_emoji = "✅" if 200 <= status_code < 300 else "⚠️" if 300 <= status_code < 400 else "❌"
            logger.info(
                f"[请求完成] {status_emoji} {request.method} {request.url.path} | "
                f"状态码: {status_code} | 处理时间: {process_time:.3f}s | 客户端: {client_ip}"
            )
            
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"[请求异常] ❌ {request.method} {request.url.path} | "
                f"错误: {str(e)} | 处理时间: {process_time:.3f}s | 客户端: {client_ip}",
                exc_info=True
            )
            raise

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_converter = None
_embedder: SentenceTransformer | None = None
_reranker = None


def _get_converter() -> PdfConverter:
    global _converter
    if _converter is None:
        device = _get_device()
        logger.info(f"Initializing marker-pdf models on GPU server... | device={device}")
        
        if device == "cuda":
            logger.info(f"使用GPU加速PDF转换 | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
        else:
            logger.warning("使用CPU运行PDF转换（性能较慢）")
        
        models = create_model_dict()
        use_llm_env = os.getenv("MARKER_USE_LLM", "false").lower()
        use_llm = use_llm_env in {"1", "true", "yes"}
        pdftext_workers = int(os.getenv("PDFTEXT_WORKERS", "1"))

        _converter = PdfConverter(
            config={
                "pdftext_workers": pdftext_workers,
                "use_llm": use_llm,
            },
            artifact_dict=models,
            processor_list=None,
            renderer=None,
            llm_service=None,
        )
        logger.info(f"marker-pdf initialized successfully on GPU server | device={device}")
    return _converter


def _get_device() -> str:
    """智能检测并返回最佳设备（GPU优先）"""
    # 检查是否强制使用CPU
    force_cpu = os.getenv("FORCE_CPU", "0").lower() in {"1", "true", "yes"}
    if force_cpu:
        return "cpu"
    
    # 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    
    # 如果设置了CUDA_VISIBLE_DEVICES且不为空，使用CUDA
    cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None and cuda_visible.strip() != "":
        if cuda_available:
            return "cuda"
        else:
            logger.warning(f"CUDA_VISIBLE_DEVICES设置为'{cuda_visible}'但CUDA不可用，使用CPU")
            return "cpu"
    
    # 如果强制使用CUDA
    force_cuda = os.getenv("FORCE_CUDA", "0").lower() in {"1", "true", "yes"}
    if force_cuda:
        if cuda_available:
            return "cuda"
        else:
            logger.warning("FORCE_CUDA=1但CUDA不可用，使用CPU")
            return "cpu"
    
    # 自动检测：如果CUDA可用，优先使用GPU
    if cuda_available:
        device_count = torch.cuda.device_count()
        if device_count > 0:
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"自动检测到GPU: {device_name} (设备数量: {device_count})")
            return "cuda"
    
    return "cpu"


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        model_name = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-large-zh-v1.5")
        device = _get_device()
        logger.info(f"Initializing embedding model on GPU server: {model_name}, device={device}")
        
        if device == "cuda":
            logger.info(f"使用GPU加速嵌入模型 | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
        else:
            logger.warning("使用CPU运行嵌入模型（性能较慢）")
        
        # 在离线模式下，使用本地缓存路径
        if os.getenv("HF_HUB_OFFLINE") == "1":
            # 尝试从本地缓存加载
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            logger.info(f"离线模式：尝试从本地缓存加载模型: {model_name}")
        
        _embedder = SentenceTransformer(model_name, device=device)
        
        # 验证实际使用的设备
        if hasattr(_embedder, '_modules') and len(_embedder._modules) > 0:
            first_module = list(_embedder._modules.values())[0]
            if hasattr(first_module, 'device'):
                actual_device = str(first_module.device)
                logger.info(f"嵌入模型实际运行设备: {actual_device}")
        
        logger.info("Embedding model initialized successfully")
    return _embedder


def _get_reranker():
    global _reranker
    if _reranker is None:
        model_name = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
        device = _get_device()
        logger.info(f"Initializing reranker model on GPU server: {model_name}, device={device}")
        
        if device == "cuda":
            logger.info(f"使用GPU加速重排序模型 | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
        else:
            logger.warning("使用CPU运行重排序模型（性能较慢）")
        
        try:
            from FlagEmbedding import FlagReranker

            use_fp16 = device == "cuda" and torch.cuda.is_available()
            _reranker = FlagReranker(model_name, use_fp16=use_fp16, device=device)
            logger.info(f"FlagReranker initialized successfully | device={device}, fp16={use_fp16}")
        except ImportError:
            logger.warning("FlagEmbedding not available, falling back to CrossEncoder")
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device=device)
            logger.info(f"CrossEncoder initialized successfully | device={device}")
    return _reranker


class EmbedRequest(BaseModel):
    texts: List[str]


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]


class RerankRequest(BaseModel):
    query: str
    documents: List[str]


class RerankResponse(BaseModel):
    scores: List[float]


@app.get("/health")
async def health_check():
    """健康检查接口"""
    logger.debug("[健康检查] 收到健康检查请求")
    return {"status": "ok"}


@app.post("/pdf_to_markdown")
async def pdf_to_markdown(file: UploadFile = File(...)):
    """Convert uploaded PDF to Markdown using marker-pdf on this server."""
    task_start_time = time.time()
    file_size = 0
    tmp_path = None
    
    try:
        logger.info(f"[PDF转换任务] 开始处理文件: {file.filename}")
        
        if not file.filename.lower().endswith(".pdf"):
            logger.warning(f"[PDF转换任务] 文件格式错误: {file.filename}")
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        # Save upload to a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            file_size = len(content)
            
            if not content:
                logger.error(f"[PDF转换任务] 文件为空: {file.filename}")
                raise HTTPException(status_code=400, detail="Empty PDF file")
            
            tmp.write(content)
            logger.info(f"[PDF转换任务] 文件已保存 | 文件名: {file.filename} | 大小: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")

        # 加载转换器
        load_start = time.time()
        converter = _get_converter()
        load_time = time.time() - load_start
        if load_time > 0.1:
            logger.info(f"[PDF转换任务] 转换器加载时间: {load_time:.3f}s")

        # 执行转换
        convert_start = time.time()
        logger.info(f"[PDF转换任务] 开始PDF转换: {file.filename}")
        rendered = converter(tmp_path)
        convert_time = time.time() - convert_start
        logger.info(f"[PDF转换任务] PDF转换完成 | 转换时间: {convert_time:.3f}s")

        # 提取文本
        extract_start = time.time()
        markdown, _, _ = text_from_rendered(rendered)
        extract_time = time.time() - extract_start
        markdown_size = len(markdown) if markdown else 0
        
        if not markdown:
            logger.error(f"[PDF转换任务] 转换结果为空: {file.filename}")
            raise HTTPException(status_code=500, detail="Empty Markdown output from marker-pdf")

        total_time = time.time() - task_start_time
        logger.info(
            f"[PDF转换任务] ✅ 任务完成 | 文件名: {file.filename} | "
            f"输入大小: {file_size:,} bytes | 输出大小: {markdown_size:,} chars | "
            f"总耗时: {total_time:.3f}s (转换: {convert_time:.3f}s, 提取: {extract_time:.3f}s)"
        )

        return {
            "content": markdown,
            "conversion_method": "marker-pdf",
            "file_name": file.filename,
        }

    except HTTPException:
        raise
    except Exception as e:
        total_time = time.time() - task_start_time
        logger.error(
            f"[PDF转换任务] ❌ 转换失败 | 文件名: {file.filename} | "
            f"文件大小: {file_size:,} bytes | 耗时: {total_time:.3f}s | 错误: {str(e)}",
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Conversion failed: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.debug(f"[PDF转换任务] 临时文件已删除: {tmp_path}")
            except Exception as e:
                logger.warning(f"[PDF转换任务] 删除临时文件失败: {tmp_path}, 错误: {e}")


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest) -> EmbedResponse:
    """Embed a batch of texts using a shared SentenceTransformer model on this server."""
    task_start_time = time.time()
    
    if not request.texts:
        logger.warning("[嵌入任务] 请求文本列表为空")
        return EmbedResponse(embeddings=[])

    try:
        text_count = len(request.texts)
        total_chars = sum(len(text) for text in request.texts)
        logger.info(
            f"[嵌入任务] 开始处理 | 文本数量: {text_count} | "
            f"总字符数: {total_chars:,} | 平均长度: {total_chars//text_count if text_count > 0 else 0:,} chars"
        )
        
        # 加载模型
        load_start = time.time()
        model = _get_embedder()
        load_time = time.time() - load_start
        if load_time > 0.1:
            logger.info(f"[嵌入任务] 模型加载时间: {load_time:.3f}s")
        
        # 执行嵌入
        encode_start = time.time()
        vectors = model.encode(request.texts, normalize_embeddings=True)
        encode_time = time.time() - encode_start
        embeddings = vectors.tolist()
        
        embedding_dim = len(embeddings[0]) if embeddings else 0
        total_time = time.time() - task_start_time
        
        logger.info(
            f"[嵌入任务] ✅ 任务完成 | 文本数量: {text_count} | "
            f"嵌入维度: {embedding_dim} | 编码时间: {encode_time:.3f}s | 总耗时: {total_time:.3f}s | "
            f"速度: {text_count/encode_time:.1f} texts/s"
        )
        
        return EmbedResponse(embeddings=embeddings)
    except Exception as e:
        total_time = time.time() - task_start_time
        logger.error(
            f"[嵌入任务] ❌ 嵌入失败 | 文本数量: {len(request.texts)} | "
            f"耗时: {total_time:.3f}s | 错误: {str(e)}",
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest) -> RerankResponse:
    """Rerank documents for a query using a shared reranker model on this server."""
    task_start_time = time.time()
    
    if not request.documents:
        logger.warning("[重排序任务] 文档列表为空")
        return RerankResponse(scores=[])

    try:
        doc_count = len(request.documents)
        query_len = len(request.query)
        total_doc_chars = sum(len(doc or "") for doc in request.documents)
        logger.info(
            f"[重排序任务] 开始处理 | 查询长度: {query_len:,} chars | "
            f"文档数量: {doc_count} | 总文档字符数: {total_doc_chars:,}"
        )
        
        # 加载模型
        load_start = time.time()
        reranker = _get_reranker()
        load_time = time.time() - load_start
        if load_time > 0.1:
            logger.info(f"[重排序任务] 模型加载时间: {load_time:.3f}s")
        
        # 准备查询-文档对
        pairs = [[request.query, doc or ""] for doc in request.documents]
        
        # 执行重排序
        rerank_start = time.time()
        # FlagReranker uses compute_score, CrossEncoder uses predict
        if hasattr(reranker, "compute_score"):
            scores = reranker.compute_score(pairs)
        else:
            scores = reranker.predict(pairs)
        rerank_time = time.time() - rerank_start

        # Ensure we return a plain list of floats
        try:
            scores_list = list(scores)
        except TypeError:
            scores_list = [scores]
        
        scores_list = [float(s) for s in scores_list]
        max_score = max(scores_list) if scores_list else 0
        min_score = min(scores_list) if scores_list else 0
        total_time = time.time() - task_start_time
        
        logger.info(
            f"[重排序任务] ✅ 任务完成 | 文档数量: {doc_count} | "
            f"重排序时间: {rerank_time:.3f}s | 总耗时: {total_time:.3f}s | "
            f"分数范围: [{min_score:.4f}, {max_score:.4f}] | "
            f"速度: {doc_count/rerank_time:.1f} pairs/s"
        )

        return RerankResponse(scores=scores_list)
    except Exception as e:
        total_time = time.time() - task_start_time
        logger.error(
            f"[重排序任务] ❌ 重排序失败 | 文档数量: {len(request.documents)} | "
            f"耗时: {total_time:.3f}s | 错误: {str(e)}",
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Rerank failed: {e}")
