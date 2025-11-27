import logging
import os
import tempfile
import time
from datetime import datetime
from typing import List
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
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
    # 禁用遥测和网络请求
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    # 确保 transformers 库完全离线
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    # 在离线模式下，monkey patch huggingface_hub 的 model_info 函数
    # 以避免 transformers 库在检查模型信息时触发网络请求
    try:
        import huggingface_hub
        from huggingface_hub import hf_api
        
        # 保存原始函数
        _original_model_info = hf_api.HfApi.model_info
        
        def _offline_model_info(self, repo_id, *args, **kwargs):
            """离线模式下的 model_info，返回一个模拟的模型信息对象"""
            from huggingface_hub.hf_api import ModelInfo
            # 返回一个基本的 ModelInfo 对象，避免网络请求
            # 这可能会在某些情况下失败，但至少不会触发网络请求
            raise RuntimeError(
                f"离线模式已启用，无法获取模型信息: {repo_id}\n"
                f"请确保模型已完全下载到本地缓存，或设置 HF_HUB_OFFLINE=0 以允许网络访问。"
            )
        
        # 替换函数
        hf_api.HfApi.model_info = _offline_model_info
    except Exception:
        # 如果 monkey patch 失败，继续使用环境变量方式
        pass
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
    # 超时阈值（秒）
    WARNING_TIMEOUT = 60  # 超过60秒发出警告
    CRITICAL_TIMEOUT = 300  # 超过300秒发出严重警告
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"
        
        # 记录请求开始
        logger.info(f"[请求开始] {request.method} {request.url.path} | 客户端: {client_ip}")
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # 超时检测和警告
            if process_time > self.CRITICAL_TIMEOUT:
                logger.warning(
                    f"[请求超时警告] ⚠️⚠️⚠️ {request.method} {request.url.path} | "
                    f"处理时间过长: {process_time:.3f}s (超过{self.CRITICAL_TIMEOUT}s) | 客户端: {client_ip}"
                )
            elif process_time > self.WARNING_TIMEOUT:
                logger.warning(
                    f"[请求超时警告] ⚠️ {request.method} {request.url.path} | "
                    f"处理时间较长: {process_time:.3f}s (超过{self.WARNING_TIMEOUT}s) | 客户端: {client_ip}"
                )
            
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
            # 确保异常被正确传播，让全局异常处理器处理
            raise

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局异常处理器 - 确保所有异常都能返回HTTP响应
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器，确保所有异常都能返回HTTP响应而不是断开连接"""
    logger.error(
        f"[全局异常] ❌ {request.method} {request.url.path} | "
        f"未捕获的异常: {type(exc).__name__}: {str(exc)}",
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Internal server error: {str(exc)}",
            "error_type": type(exc).__name__
        }
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """HTTP异常处理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

_converter = None
_embedder: SentenceTransformer | None = None
_reranker = None
_ocr_engine = None


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
            
            # 在离线模式下，尝试从本地缓存加载
            if os.getenv("HF_HUB_OFFLINE") == "1":
                # 尝试获取本地缓存路径
                cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                model_cache = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
                
                if os.path.exists(model_cache):
                    # 查找最新的快照
                    snapshots_dir = os.path.join(model_cache, "snapshots")
                    if os.path.exists(snapshots_dir):
                        snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                        if snapshots:
                            # 使用最新的快照
                            local_path = os.path.join(snapshots_dir, snapshots[-1])
                            logger.info(f"离线模式：从本地缓存加载模型: {local_path}")
                            try:
                                _reranker = FlagReranker(local_path, use_fp16=use_fp16, device=device)
                                logger.info(f"FlagReranker initialized successfully from local cache | device={device}, fp16={use_fp16}")
                                return _reranker
                            except Exception as e:
                                error_msg = (
                                    f"离线模式下从本地路径加载模型失败: {e}\n"
                                    f"本地路径: {local_path}\n"
                                    f"请检查模型文件是否完整，或设置 HF_HUB_OFFLINE=0 以允许网络访问。"
                                )
                                logger.error(error_msg)
                                raise RuntimeError(error_msg) from e
                
                # 离线模式下未找到本地缓存
                error_msg = (
                    f"离线模式下未找到模型 {model_name} 的本地缓存。\n"
                    f"缓存目录: {cache_dir}\n"
                    f"请先下载模型，或设置 HF_HUB_OFFLINE=0 以允许网络访问。\n"
                    f"下载命令: python -c \"from huggingface_hub import snapshot_download; snapshot_download('{model_name}')\""
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # 在线模式：正常加载
            _reranker = FlagReranker(model_name, use_fp16=use_fp16, device=device)
            logger.info(f"FlagReranker initialized successfully | device={device}, fp16={use_fp16}")
        except ImportError:
            logger.warning("FlagEmbedding not available, falling back to CrossEncoder")
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device=device)
            logger.info(f"CrossEncoder initialized successfully | device={device}")
        except Exception as e:
            if os.getenv("HF_HUB_OFFLINE") == "1":
                error_msg = (
                    f"离线模式下加载重排序模型失败: {e}\n"
                    f"请确保模型 {model_name} 已完全下载到本地缓存。\n"
                    f"可以运行以下命令下载模型：\n"
                    f"  python -c \"from huggingface_hub import snapshot_download; snapshot_download('{model_name}')\"\n"
                    f"或者设置 HF_HUB_OFFLINE=0 以允许网络访问。"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
            else:
                raise
    return _reranker


def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from paddleocr import PaddleOCR
            import paddle
        except ImportError as e:
            logger.error("PaddleOCR not available on GPU server: %s", e)
            raise

        # 检查 GPU 配置
        use_gpu_env = os.getenv("GPU_OCR_USE_GPU", "1").lower()
        use_gpu = use_gpu_env in {"1", "true", "yes"}
        
        # 新版本的 PaddleOCR 不再支持 use_gpu 参数
        # GPU 使用由 PaddlePaddle 自动检测，或通过环境变量控制
        if use_gpu:
            # 设置 PaddlePaddle 使用 GPU
            try:
                if paddle.device.is_compiled_with_cuda():
                    # 设置默认设备为 GPU
                    paddle.set_device('gpu')
                    logger.info("OCR engine will use GPU (PaddlePaddle CUDA enabled)")
                else:
                    logger.warning("PaddlePaddle 未编译 CUDA 支持，OCR 将使用 CPU")
                    use_gpu = False
            except Exception as e:
                logger.warning(f"设置 GPU 设备失败: {e}，将使用 CPU")
                use_gpu = False
        else:
            paddle.set_device('cpu')
            logger.info("OCR engine will use CPU (GPU_OCR_USE_GPU=0)")

        # 初始化 PaddleOCR（新版本 API）
        try:
            _ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang="ch",
            )
            logger.info("OCR engine initialized with PaddleOCR (use_gpu=%s)", use_gpu)
        except Exception as e:
            # 如果失败，尝试更简单的初始化
            logger.warning(f"使用标准参数初始化失败: {e}，尝试简化初始化")
            _ocr_engine = PaddleOCR(lang="ch")
            logger.info("OCR engine initialized with PaddleOCR (simplified, use_gpu=%s)", use_gpu)
    return _ocr_engine


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
        try:
            model = _get_embedder()
        except Exception as model_error:
            logger.error(f"[嵌入任务] 模型加载失败: {str(model_error)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to load embedding model: {model_error}")
        
        load_time = time.time() - load_start
        if load_time > 0.1:
            logger.info(f"[嵌入任务] 模型加载时间: {load_time:.3f}s")
        
        # 执行嵌入
        encode_start = time.time()
        try:
            vectors = model.encode(request.texts, normalize_embeddings=True)
        except Exception as encode_error:
            logger.error(f"[嵌入任务] 编码过程失败: {str(encode_error)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Encoding failed: {encode_error}")
        
        encode_time = time.time() - encode_start
        
        # 安全转换为列表
        try:
            embeddings = vectors.tolist()
        except Exception as convert_error:
            logger.error(f"[嵌入任务] 向量转换失败: {str(convert_error)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Vector conversion failed: {convert_error}")
        
        embedding_dim = len(embeddings[0]) if embeddings else 0
        total_time = time.time() - task_start_time
        
        logger.info(
            f"[嵌入任务] ✅ 任务完成 | 文本数量: {text_count} | "
            f"嵌入维度: {embedding_dim} | 编码时间: {encode_time:.3f}s | 总耗时: {total_time:.3f}s | "
            f"速度: {text_count/encode_time:.1f} texts/s"
        )
        
        return EmbedResponse(embeddings=embeddings)
    except HTTPException:
        # 重新抛出HTTP异常，确保响应被发送
        raise
    except Exception as e:
        total_time = time.time() - task_start_time
        logger.error(
            f"[嵌入任务] ❌ 嵌入失败 | 文本数量: {len(request.texts) if request.texts else 0} | "
            f"耗时: {total_time:.3f}s | 错误: {str(e)}",
            exc_info=True
        )
        # 确保返回HTTP响应而不是让连接断开
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")


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


class OCRResponse(BaseModel):
    text: str
    confidence: float
    lines: List[str]
    confidences: List[float]
    boxes: List[List[List[float]]]


@app.post("/ocr_image", response_model=OCRResponse)
async def ocr_image(file: UploadFile = File(...)) -> OCRResponse:
    task_start_time = time.time()
    tmp_path = None

    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Empty filename")

        suffix = os.path.splitext(file.filename)[1].lower() or ".png"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Empty image file")
            tmp.write(content)

        ocr_engine = _get_ocr_engine()

        ocr_result = ocr_engine.ocr(tmp_path, cls=True)

        lines: List[str] = []
        confidences: List[float] = []
        boxes: List[List[List[float]]] = []

        if ocr_result and ocr_result[0]:
            for item in ocr_result[0]:
                if not item or len(item) < 2:
                    continue
                box = item[0]
                text_info = item[1]
                if not text_info or len(text_info) < 2:
                    continue
                text = text_info[0]
                score = float(text_info[1])
                if text:
                    lines.append(text)
                    confidences.append(score)
                    boxes.append(box)

        full_text = "\n".join(lines)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        elapsed = time.time() - task_start_time

        logger.info(
            "[OCR任务] 完成 | 文件名: %s | 文本长度: %d | 置信度: %.4f | 耗时: %.3fs",
            file.filename,
            len(full_text),
            avg_conf,
            elapsed,
        )

        return OCRResponse(
            text=full_text,
            confidence=avg_conf,
            lines=lines,
            confidences=confidences,
            boxes=boxes,
        )
    except HTTPException:
        raise
    except Exception as e:
        elapsed = time.time() - task_start_time
        logger.error(
            "[OCR任务] 失败 | 文件名: %s | 耗时: %.3fs | 错误: %s",
            file.filename,
            elapsed,
            str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
