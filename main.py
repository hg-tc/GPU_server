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

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

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
    
    # 降低第三方库的日志级别
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    return logger

logger = setup_logging()

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
        logger.info("Initializing marker-pdf models on GPU server...")
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
        logger.info("marker-pdf initialized successfully on GPU server")
    return _converter


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        model_name = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-large-zh-v1.5")
        device = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") or os.getenv("FORCE_CUDA", "0") == "1" else "cpu"
        logger.info(f"Initializing embedding model on GPU server: {model_name}, device={device}")
        _embedder = SentenceTransformer(model_name, device=device)
        logger.info("Embedding model initialized successfully")
    return _embedder


def _get_reranker():
    global _reranker
    if _reranker is None:
        model_name = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
        device = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") or os.getenv("FORCE_CUDA", "0") == "1" else "cpu"
        logger.info(f"Initializing reranker model on GPU server: {model_name}, device={device}")
        try:
            from FlagEmbedding import FlagReranker

            use_fp16 = device == "cuda"
            _reranker = FlagReranker(model_name, use_fp16=use_fp16, device=device)
            logger.info("FlagReranker initialized successfully")
        except ImportError:
            logger.warning("FlagEmbedding not available, falling back to CrossEncoder")
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device=device)
            logger.info("CrossEncoder initialized successfully")
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
