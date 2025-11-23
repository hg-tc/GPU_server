import logging
import os
import tempfile
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# 配置 Hugging Face 镜像源（如果未设置）
if not os.getenv("HF_ENDPOINT"):
    # 使用 hf-mirror.com 作为默认镜像源
    os.environ["HF_ENDPOINT"] = os.getenv("HF_MIRROR_ENDPOINT", "https://hf-mirror.com")

logger = logging.getLogger("gpu_pdf_server")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="GPU Model Server",
    version="0.1.0",
    description="Offload marker-pdf PDF->Markdown, embeddings, and reranking to a dedicated server",
)

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
    return {"status": "ok"}


@app.post("/pdf_to_markdown")
async def pdf_to_markdown(file: UploadFile = File(...)):
    """Convert uploaded PDF to Markdown using marker-pdf on this server."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    tmp_path = None
    try:
        # Save upload to a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Empty PDF file")
            tmp.write(content)

        converter = _get_converter()
        rendered = converter(tmp_path)
        markdown, _, _ = text_from_rendered(rendered)

        if not markdown:
            raise HTTPException(status_code=500, detail="Empty Markdown output from marker-pdf")

        return {
            "content": markdown,
            "conversion_method": "marker-pdf",
            "file_name": file.filename,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("PDF conversion failed on GPU server")
        raise HTTPException(status_code=500, detail=f"Conversion failed: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest) -> EmbedResponse:
    """Embed a batch of texts using a shared SentenceTransformer model on this server."""
    if not request.texts:
        return EmbedResponse(embeddings=[])

    try:
        model = _get_embedder()
        vectors = model.encode(request.texts, normalize_embeddings=True)
        embeddings = vectors.tolist()
        return EmbedResponse(embeddings=embeddings)
    except Exception as e:
        logger.exception("Embedding failed on GPU server")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest) -> RerankResponse:
    """Rerank documents for a query using a shared reranker model on this server."""
    if not request.documents:
        return RerankResponse(scores=[])

    try:
        reranker = _get_reranker()
        pairs = [[request.query, doc or ""] for doc in request.documents]

        # FlagReranker uses compute_score, CrossEncoder uses predict
        if hasattr(reranker, "compute_score"):
            scores = reranker.compute_score(pairs)
        else:
            scores = reranker.predict(pairs)

        # Ensure we return a plain list of floats
        try:
            scores_list = list(scores)
        except TypeError:
            scores_list = [scores]

        return RerankResponse(scores=[float(s) for s in scores_list])
    except Exception as e:
        logger.exception("Rerank failed on GPU server")
        raise HTTPException(status_code=500, detail=f"Rerank failed: {e}")
