"""
GPU Model Server - PaddleOCR 3.x Version
=========================================
支持:
- PDF 转 Markdown (PP-StructureV3)
- 图片 OCR (PaddleOCR 3.x)
- 文档版面分析 (PP-StructureV3)
- 文本嵌入 (sentence-transformers)
- 文本重排序 (FlagEmbedding)
"""

import logging
import os
import tempfile
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
import base64
import io

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
import torch

# ============================================
# 日志配置
# ============================================
def setup_logging():
    """配置详细的日志系统"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_format = "%(asctime)s [%(levelname)-8s] [%(name)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # 主日志文件处理器（启用实时刷新）
    file_handler = logging.FileHandler(
        log_dir / "gpu_server.log",
        encoding="utf-8",
        mode="a"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    # 设置延迟为 False，每次写入后立即刷新
    if hasattr(file_handler, 'stream'):
        file_handler.stream.reconfigure(line_buffering=True)
    
    # 错误日志文件处理器（启用实时刷新）
    error_handler = logging.FileHandler(
        log_dir / "gpu_server_error.log",
        encoding="utf-8",
        mode="a"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(log_format, date_format))
    # 设置延迟为 False，每次写入后立即刷新
    if hasattr(error_handler, 'stream'):
        error_handler.stream.reconfigure(line_buffering=True)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)
    
    # 为文件处理器添加自动刷新机制
    for handler in [file_handler, error_handler]:
        original_emit = handler.emit
        def make_flush_emit(h):
            def flush_emit(record):
                result = original_emit(record)
                if hasattr(h, 'stream') and hasattr(h.stream, 'flush'):
                    h.stream.flush()
                return result
            return flush_emit
        handler.emit = make_flush_emit(handler)
    
    logger = logging.getLogger("gpu_server")
    logger.setLevel(logging.DEBUG)
    
    # 降低第三方库日志级别
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("paddle").setLevel(logging.WARNING)
    
    return logger

logger = setup_logging()

# ============================================
# GPU 检测
# ============================================
def detect_device():
    """检测可用设备"""
    # 检查 PaddlePaddle GPU
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda():
            gpu_count = paddle.device.cuda.device_count()
            if gpu_count > 0:
                logger.info("=" * 60)
                logger.info("🚀 PaddlePaddle GPU 检测:")
                logger.info(f"  ✅ CUDA 可用")
                logger.info(f"  🎮 GPU 数量: {gpu_count}")
                for i in range(gpu_count):
                    props = paddle.device.cuda.get_device_properties(i)
                    logger.info(f"  🎯 GPU {i}: {props.name}")
                logger.info("=" * 60)
                return "gpu"
    except Exception as e:
        logger.warning(f"PaddlePaddle GPU 检测失败: {e}")
    
    # 检查 PyTorch GPU
    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info("=" * 60)
            logger.info("🚀 PyTorch GPU 检测:")
            logger.info(f"  ✅ CUDA 可用")
            logger.info(f"  🎮 GPU 数量: {gpu_count}")
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  🎯 GPU {i}: {props.name}")
                logger.info(f"     显存: {props.total_memory / 1024**3:.2f} GB")
            logger.info("=" * 60)
            return "cuda"
    except Exception as e:
        logger.warning(f"PyTorch GPU 检测失败: {e}")
    
    logger.warning("⚠️ 未检测到 GPU，将使用 CPU")
    return "cpu"

DEVICE = detect_device()

# ============================================
# FastAPI 应用
# ============================================
app = FastAPI(
    title="GPU Model Server - PaddleOCR 3.x",
    version="3.0.0",
    description="PDF/图片处理、OCR、嵌入、重排序服务",
)

# 请求日志中间件
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    WARNING_TIMEOUT = 60
    CRITICAL_TIMEOUT = 300
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"
        
        logger.info(f"[请求开始] {request.method} {request.url.path} | 客户端: {client_ip}")
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            if process_time > self.CRITICAL_TIMEOUT:
                logger.warning(f"[请求超时] ⚠️⚠️⚠️ {request.url.path} | 耗时: {process_time:.3f}s")
            elif process_time > self.WARNING_TIMEOUT:
                logger.warning(f"[请求超时] ⚠️ {request.url.path} | 耗时: {process_time:.3f}s")
            
            status_emoji = "✅" if 200 <= response.status_code < 300 else "❌"
            logger.info(f"[请求完成] {status_emoji} {request.url.path} | 状态: {response.status_code} | 耗时: {process_time:.3f}s")
            
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"[请求异常] ❌ {request.url.path} | 错误: {e} | 耗时: {process_time:.3f}s", exc_info=True)
            raise

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"[全局异常] {request.url.path} | {type(exc).__name__}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}", "error_type": type(exc).__name__}
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

# ============================================
# 模型单例管理
# ============================================
_ocr_engine = None
_structure_engine = None
_embedder = None
_reranker = None

def _get_ocr_engine():
    """获取 PaddleOCR 3.x 引擎"""
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from paddleocr import PaddleOCR
            logger.info("初始化 PaddleOCR 3.x...")
            
            device = "gpu" if DEVICE in ["gpu", "cuda"] else "cpu"
            _ocr_engine = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device=device,
            )
            logger.info(f"PaddleOCR 3.x 初始化成功 | device={device}")
        except Exception as e:
            logger.error(f"PaddleOCR 初始化失败: {e}")
            raise
    return _ocr_engine


def _get_structure_engine():
    """获取 PP-StructureV3 引擎"""
    global _structure_engine
    if _structure_engine is None:
        try:
            from paddleocr import PPStructureV3
            logger.info("初始化 PP-StructureV3...")
            
            device = "gpu" if DEVICE in ["gpu", "cuda"] else "cpu"
            _structure_engine = PPStructureV3(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                device=device,
            )
            logger.info(f"PP-StructureV3 初始化成功 | device={device}")
        except Exception as e:
            logger.error(f"PP-StructureV3 初始化失败: {e}")
            raise
    return _structure_engine


def _get_embedder():
    """获取嵌入模型"""
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            
            # 设置 HuggingFace 离线模式（使用本地缓存）
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            
            model_name = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-large-zh-v1.5")
            device = "cuda" if DEVICE in ["gpu", "cuda"] else "cpu"
            
            logger.info(f"初始化嵌入模型: {model_name} | device={device} | 离线模式（仅本地文件）")
            # 使用 local_files_only=True 强制只使用本地缓存
            _embedder = SentenceTransformer(model_name, device=device, local_files_only=True)
            logger.info("嵌入模型初始化成功")
        except Exception as e:
            logger.error(f"嵌入模型初始化失败: {e}")
            raise
    return _embedder


def _get_reranker():
    """获取重排序模型"""
    global _reranker
    if _reranker is None:
        try:
            from FlagEmbedding import FlagReranker
            import glob
            
            # 设置 HuggingFace 离线模式（使用本地缓存）
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            
            model_name = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
            device = "cuda" if DEVICE in ["gpu", "cuda"] else "cpu"
            use_fp16 = device == "cuda"
            
            # 尝试找到本地缓存的模型路径
            hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            cache_dir = os.path.join(hf_home, "hub")
            model_cache_name = f"models--{model_name.replace('/', '--')}"
            model_cache_path = os.path.join(cache_dir, model_cache_name, "snapshots")
            
            # 查找最新的快照目录
            local_model_path = None
            if os.path.exists(model_cache_path):
                snapshots = sorted(glob.glob(os.path.join(model_cache_path, "*")), reverse=True)
                if snapshots and os.path.isdir(snapshots[0]):
                    local_model_path = snapshots[0]
                    logger.info(f"找到本地模型缓存: {local_model_path}")
            
            # 如果找到本地路径，使用本地路径；否则使用模型名称（依赖环境变量）
            if local_model_path:
                logger.info(f"初始化重排序模型: {local_model_path} | device={device} | 离线模式（本地路径）")
                _reranker = FlagReranker(
                    local_model_path,
                    use_fp16=use_fp16,
                    devices=device
                )
            else:
                logger.info(f"初始化重排序模型: {model_name} | device={device} | 离线模式（使用缓存）")
                _reranker = FlagReranker(
                    model_name,
                    use_fp16=use_fp16,
                    devices=device,
                    cache_dir=cache_dir
                )
            logger.info("重排序模型初始化成功")
        except ImportError:
            from sentence_transformers import CrossEncoder
            logger.warning("FlagEmbedding 不可用，使用 CrossEncoder")
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device=device)
        except Exception as e:
            logger.error(f"重排序模型初始化失败: {e}")
            raise
    return _reranker

# ============================================
# 统计信息
# ============================================
_server_start_time = time.time()
_request_stats = {
    "ocr_total": 0, "ocr_success": 0, "ocr_failed": 0,
    "structure_total": 0, "structure_success": 0, "structure_failed": 0,
    "pdf_total": 0, "pdf_success": 0, "pdf_failed": 0,
    "embed_total": 0, "embed_success": 0, "embed_failed": 0,
    "rerank_total": 0, "rerank_success": 0, "rerank_failed": 0,
}

# ============================================
# API 端点
# ============================================

@app.get("/health")
async def health_check():
    """健康检查"""
    status = {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "device": DEVICE,
    }
    
    # GPU 信息
    try:
        if torch.cuda.is_available():
            status["gpu"] = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "memory": {
                    "allocated_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
                    "reserved_gb": round(torch.cuda.memory_reserved(0) / 1024**3, 2),
                    "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
                }
            }
        else:
            status["gpu"] = {"available": False}
    except Exception as e:
        status["gpu"] = {"available": False, "error": str(e)}
    
    # 模型状态（懒加载：只有在使用时才会加载）
    status["models"] = {
        "ocr": "loaded" if _ocr_engine is not None else "lazy",
        "structure": "loaded" if _structure_engine is not None else "lazy",
        "embedder": "loaded" if _embedder is not None else "lazy",
        "reranker": "loaded" if _reranker is not None else "lazy",
    }
    
    return status


@app.get("/stats")
async def get_stats():
    """获取统计信息"""
    return {
        "stats": _request_stats,
        "uptime_seconds": time.time() - _server_start_time,
    }


@app.post("/clear_cache")
async def clear_gpu_cache():
    """清理 GPU 缓存"""
    try:
        if torch.cuda.is_available():
            before = torch.cuda.memory_allocated(0) / 1024**3
            torch.cuda.empty_cache()
            after = torch.cuda.memory_allocated(0) / 1024**3
            return {"status": "ok", "freed_gb": round(before - after, 2)}
        return {"status": "ok", "message": "No GPU"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# OCR 端点
# ============================================

class OCRResponse(BaseModel):
    text: str
    confidence: float
    lines: List[str]
    confidences: List[float]
    boxes: List[List[List[float]]]


@app.post("/ocr_image", response_model=OCRResponse)
async def ocr_image(file: UploadFile = File(...)) -> OCRResponse:
    """图片 OCR - 使用 PaddleOCR 3.x"""
    global _request_stats
    _request_stats["ocr_total"] += 1
    
    start_time = time.time()
    tmp_path = None
    
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Empty filename")
        
        suffix = os.path.splitext(file.filename)[1].lower() or ".png"
        
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Empty file")
            tmp.write(content)
            tmp.flush()  # 确保数据写入磁盘
            os.fsync(tmp.fileno())  # 强制同步到磁盘
        
        ocr = _get_ocr_engine()
        
        # PaddleOCR 3.x 使用 predict 方法
        results = ocr.predict(tmp_path)
        
        lines = []
        confidences = []
        boxes = []
        
        # 调试：记录原始结果类型和内容
        logger.debug(f"[OCR] 结果类型: {type(results)}, 结果数量: {len(results) if hasattr(results, '__len__') else 'N/A'}")
        if isinstance(results, list) and len(results) > 0:
            logger.debug(f"[OCR] 第一个结果类型: {type(results[0])}, 内容: {str(results[0])[:200]}")
            if hasattr(results[0], '__dict__'):
                logger.debug(f"[OCR] 第一个结果属性: {list(results[0].__dict__.keys())}")
        
        # 处理结果 - PaddleOCR 3.x 可能返回不同的格式
        if isinstance(results, list):
            for idx, res in enumerate(results):
                logger.debug(f"[OCR] 处理结果 {idx}: 类型={type(res)}")
                # 方法1: 检查是否有 rec_texts 属性
                if hasattr(res, 'rec_texts'):
                    rec_texts = res.rec_texts if hasattr(res, 'rec_texts') else []
                    rec_scores = res.rec_scores if hasattr(res, 'rec_scores') else []
                    rec_polys = res.rec_polys if hasattr(res, 'rec_polys') else []
                    
                    for i, text in enumerate(rec_texts):
                        if text and text.strip():
                            lines.append(text.strip())
                            confidences.append(float(rec_scores[i]) if i < len(rec_scores) else 0.0)
                            if i < len(rec_polys):
                                boxes.append(rec_polys[i].tolist() if hasattr(rec_polys[i], 'tolist') else rec_polys[i])
                
                # 方法2: 检查是否有 json 属性
                elif hasattr(res, 'json'):
                    data = res.json
                    if isinstance(data, dict) and 'rec_texts' in data:
                        for i, text in enumerate(data['rec_texts']):
                            if text and text.strip():
                                lines.append(text.strip())
                                confidences.append(float(data['rec_scores'][i]) if i < len(data.get('rec_scores', [])) else 0.0)
                
                # 方法3: 直接是字典格式
                elif isinstance(res, dict):
                    if 'rec_texts' in res:
                        for i, text in enumerate(res['rec_texts']):
                            if text and text.strip():
                                lines.append(text.strip())
                                confidences.append(float(res['rec_scores'][i]) if i < len(res.get('rec_scores', [])) else 0.0)
                                if 'rec_polys' in res and i < len(res['rec_polys']):
                                    boxes.append(res['rec_polys'][i])
                
                # 方法4: 尝试直接访问文本（兼容旧版本格式）
                elif isinstance(res, (list, tuple)) and len(res) >= 2:
                    # 格式: [[[x1,y1], [x2,y2], ...], (text, confidence)]
                    for item in res:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            text = item[1][0] if isinstance(item[1], (list, tuple)) else str(item[1])
                            conf = item[1][1] if isinstance(item[1], (list, tuple)) and len(item[1]) > 1 else 0.0
                            if text and text.strip():
                                lines.append(text.strip())
                                confidences.append(float(conf))
                                if isinstance(item[0], (list, tuple)):
                                    boxes.append(item[0])
        
        logger.debug(f"[OCR] 解析结果: {len(lines)} 行文本")
        
        full_text = "\n".join(lines)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        
        elapsed = time.time() - start_time
        logger.info(f"[OCR] ✅ 完成 | 文件: {file.filename} | 文本长度: {len(full_text)} | 置信度: {avg_conf:.4f} | 耗时: {elapsed:.3f}s")
        
        _request_stats["ocr_success"] += 1
        return OCRResponse(
            text=full_text,
            confidence=avg_conf,
            lines=lines,
            confidences=confidences,
            boxes=boxes,
        )
    
    except HTTPException:
        _request_stats["ocr_failed"] += 1
        raise
    except Exception as e:
        _request_stats["ocr_failed"] += 1
        logger.error(f"[OCR] ❌ 失败 | 文件: {file.filename} | 错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass


class OCRBase64Request(BaseModel):
    image_base64: str
    filename: Optional[str] = "image.png"


@app.post("/ocr_base64", response_model=OCRResponse)
async def ocr_base64(request: OCRBase64Request) -> OCRResponse:
    """Base64 图片 OCR"""
    global _request_stats
    _request_stats["ocr_total"] += 1
    
    start_time = time.time()
    tmp_path = None
    
    try:
        # 解码 base64
        image_data = base64.b64decode(request.image_base64)
        
        suffix = os.path.splitext(request.filename)[1].lower() or ".png"
        
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(image_data)
            tmp.flush()  # 确保数据写入磁盘
            os.fsync(tmp.fileno())  # 强制同步到磁盘
        
        ocr = _get_ocr_engine()
        results = ocr.predict(tmp_path)
        
        lines = []
        confidences = []
        boxes = []
        
        # 使用与 ocr_image 相同的解析逻辑
        if isinstance(results, list):
            for idx, res in enumerate(results):
                logger.debug(f"[OCR-Base64] 处理结果 {idx}: 类型={type(res)}")
                if hasattr(res, 'rec_texts'):
                    rec_texts = res.rec_texts if hasattr(res, 'rec_texts') else []
                    rec_scores = res.rec_scores if hasattr(res, 'rec_scores') else []
                    
                    for i, text in enumerate(rec_texts):
                        if text and text.strip():
                            lines.append(text.strip())
                            confidences.append(float(rec_scores[i]) if i < len(rec_scores) else 0.0)
                elif isinstance(res, dict) and 'rec_texts' in res:
                    for i, text in enumerate(res['rec_texts']):
                        if text and text.strip():
                            lines.append(text.strip())
                            confidences.append(float(res['rec_scores'][i]) if i < len(res.get('rec_scores', [])) else 0.0)
                elif isinstance(res, (list, tuple)) and len(res) >= 2:
                    for item in res:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            text = item[1][0] if isinstance(item[1], (list, tuple)) else str(item[1])
                            conf = item[1][1] if isinstance(item[1], (list, tuple)) and len(item[1]) > 1 else 0.0
                            if text and text.strip():
                                lines.append(text.strip())
                                confidences.append(float(conf))
        
        full_text = "\n".join(lines)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        
        elapsed = time.time() - start_time
        logger.info(f"[OCR-Base64] ✅ 完成 | 文本长度: {len(full_text)} | 耗时: {elapsed:.3f}s")
        
        _request_stats["ocr_success"] += 1
        return OCRResponse(
            text=full_text,
            confidence=avg_conf,
            lines=lines,
            confidences=confidences,
            boxes=boxes,
        )
    
    except Exception as e:
        _request_stats["ocr_failed"] += 1
        logger.error(f"[OCR-Base64] ❌ 失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass


# ============================================
# 文档结构分析端点
# ============================================

class StructureResponse(BaseModel):
    markdown: str
    layout_info: Optional[Dict[str, Any]] = None


@app.post("/structure_image", response_model=StructureResponse)
async def structure_image(file: UploadFile = File(...)) -> StructureResponse:
    """图片版面分析 - 使用 PP-StructureV3"""
    global _request_stats
    _request_stats["structure_total"] += 1
    
    start_time = time.time()
    tmp_path = None
    
    try:
        suffix = os.path.splitext(file.filename)[1].lower() or ".png"
        
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)
        
        structure = _get_structure_engine()
        results = structure.predict(tmp_path)
        
        markdown_text = ""
        layout_info = {}
        
        for res in results:
            if hasattr(res, 'markdown'):
                md_info = res.markdown
                if isinstance(md_info, dict):
                    markdown_text = md_info.get('markdown_text', '')
                elif isinstance(md_info, str):
                    markdown_text = md_info
            
            if hasattr(res, 'json'):
                layout_info = res.json
        
        elapsed = time.time() - start_time
        logger.info(f"[Structure] ✅ 完成 | 文件: {file.filename} | Markdown长度: {len(markdown_text)} | 耗时: {elapsed:.3f}s")
        
        _request_stats["structure_success"] += 1
        return StructureResponse(markdown=markdown_text, layout_info=layout_info)
    
    except Exception as e:
        _request_stats["structure_failed"] += 1
        logger.error(f"[Structure] ❌ 失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Structure analysis failed: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass


@app.post("/structure_pdf")
async def structure_pdf(file: UploadFile = File(...)):
    """PDF 版面分析 - 使用 PP-StructureV3"""
    global _request_stats
    _request_stats["structure_total"] += 1
    
    start_time = time.time()
    tmp_path = None
    
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files supported")
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)
        
        structure = _get_structure_engine()
        results = structure.predict(tmp_path)
        
        # 合并所有页面的 Markdown
        markdown_list = []
        for res in results:
            if hasattr(res, 'markdown'):
                md_info = res.markdown
                markdown_list.append(md_info)
        
        # 使用 PP-StructureV3 的合并方法
        if hasattr(structure, 'concatenate_markdown_pages'):
            full_markdown = structure.concatenate_markdown_pages(markdown_list)
        else:
            full_markdown = "\n\n---\n\n".join([
                m.get('markdown_text', '') if isinstance(m, dict) else str(m)
                for m in markdown_list
            ])
        
        elapsed = time.time() - start_time
        logger.info(f"[Structure-PDF] ✅ 完成 | 文件: {file.filename} | 页数: {len(markdown_list)} | 耗时: {elapsed:.3f}s")
        
        _request_stats["structure_success"] += 1
        return {
            "content": full_markdown,
            "page_count": len(markdown_list),
            "conversion_method": "PP-StructureV3",
        }
    
    except HTTPException:
        _request_stats["structure_failed"] += 1
        raise
    except Exception as e:
        _request_stats["structure_failed"] += 1
        logger.error(f"[Structure-PDF] ❌ 失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF structure analysis failed: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass


# ============================================
# PDF 转 Markdown 端点
# ============================================

@app.post("/pdf_to_markdown")
async def pdf_to_markdown(file: UploadFile = File(...)):
    """PDF 转 Markdown (使用 PP-StructureV3)"""
    global _request_stats
    _request_stats["pdf_total"] += 1
    
    start_time = time.time()
    tmp_path = None
    
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files supported")
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            file_size = len(content)
            tmp.write(content)
        
        logger.info(f"[PDF] 开始处理 | 文件: {file.filename} | 大小: {file_size/1024/1024:.2f}MB")
        
        # 使用 PP-StructureV3
        structure = _get_structure_engine()
        results = structure.predict(tmp_path)
        
        markdown_list = []
        for res in results:
            if hasattr(res, 'markdown'):
                markdown_list.append(res.markdown)
        
        if hasattr(structure, 'concatenate_markdown_pages'):
            markdown = structure.concatenate_markdown_pages(markdown_list)
        else:
            markdown = "\n\n---\n\n".join([
                m.get('markdown_text', '') if isinstance(m, dict) else str(m)
                for m in markdown_list
            ])
        
        if not markdown:
            raise HTTPException(status_code=500, detail="Empty output")
        
        elapsed = time.time() - start_time
        logger.info(f"[PDF] ✅ 完成 | 文件: {file.filename} | 输出: {len(markdown)} chars | 耗时: {elapsed:.3f}s")
        
        _request_stats["pdf_success"] += 1
        return {
            "content": markdown,
            "conversion_method": "PP-StructureV3",
            "file_name": file.filename,
        }
    
    except HTTPException:
        _request_stats["pdf_failed"] += 1
        raise
    except Exception as e:
        _request_stats["pdf_failed"] += 1
        logger.error(f"[PDF] ❌ 失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF conversion failed: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass


# ============================================
# 嵌入端点
# ============================================

class EmbedRequest(BaseModel):
    texts: List[str]


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest) -> EmbedResponse:
    """文本嵌入"""
    global _request_stats
    _request_stats["embed_total"] += 1
    
    start_time = time.time()
    
    if not request.texts:
        return EmbedResponse(embeddings=[])
    
    try:
        model = _get_embedder()
        vectors = model.encode(request.texts, normalize_embeddings=True)
        embeddings = vectors.tolist()
        
        elapsed = time.time() - start_time
        logger.info(f"[Embed] ✅ 完成 | 文本数: {len(request.texts)} | 耗时: {elapsed:.3f}s")
        
        _request_stats["embed_success"] += 1
        return EmbedResponse(embeddings=embeddings)
    
    except Exception as e:
        _request_stats["embed_failed"] += 1
        logger.error(f"[Embed] ❌ 失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")


class BatchEmbedRequest(BaseModel):
    batches: List[List[str]]


class BatchEmbedResponse(BaseModel):
    embeddings: List[List[List[float]]]
    batch_times: List[float]


@app.post("/embed_batch", response_model=BatchEmbedResponse)
async def embed_batch(request: BatchEmbedRequest) -> BatchEmbedResponse:
    """批量文本嵌入"""
    global _request_stats
    _request_stats["embed_total"] += 1
    
    if not request.batches:
        return BatchEmbedResponse(embeddings=[], batch_times=[])
    
    try:
        model = _get_embedder()
        all_embeddings = []
        batch_times = []
        
        for batch in request.batches:
            if not batch:
                all_embeddings.append([])
                batch_times.append(0)
                continue
            
            batch_start = time.time()
            vectors = model.encode(batch, normalize_embeddings=True)
            batch_time = time.time() - batch_start
            
            all_embeddings.append(vectors.tolist())
            batch_times.append(round(batch_time, 3))
        
        _request_stats["embed_success"] += 1
        return BatchEmbedResponse(embeddings=all_embeddings, batch_times=batch_times)
    
    except Exception as e:
        _request_stats["embed_failed"] += 1
        raise HTTPException(status_code=500, detail=f"Batch embedding failed: {e}")


# ============================================
# 重排序端点
# ============================================

class RerankRequest(BaseModel):
    query: str
    documents: List[str]


class RerankResponse(BaseModel):
    scores: List[float]


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest) -> RerankResponse:
    """文档重排序"""
    global _request_stats
    _request_stats["rerank_total"] += 1
    
    start_time = time.time()
    
    if not request.documents:
        return RerankResponse(scores=[])
    
    try:
        reranker = _get_reranker()
        pairs = [[request.query, doc or ""] for doc in request.documents]
        
        if hasattr(reranker, "compute_score"):
            scores = reranker.compute_score(pairs)
        else:
            scores = reranker.predict(pairs)
        
        try:
            scores_list = list(scores)
        except TypeError:
            scores_list = [scores]
        
        scores_list = [float(s) for s in scores_list]
        
        elapsed = time.time() - start_time
        logger.info(f"[Rerank] ✅ 完成 | 文档数: {len(request.documents)} | 耗时: {elapsed:.3f}s")
        
        _request_stats["rerank_success"] += 1
        return RerankResponse(scores=scores_list)
    
    except Exception as e:
        _request_stats["rerank_failed"] += 1
        logger.error(f"[Rerank] ❌ 失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Rerank failed: {e}")


# ============================================
# 启动入口
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    logger.info(f"启动 GPU Server | host={host} | port={port}")
    uvicorn.run(app, host=host, port=port)
