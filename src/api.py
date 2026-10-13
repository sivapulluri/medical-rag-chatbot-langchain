"""FastAPI backend for the Medical RAG assistant.

Run from the project root:

    uvicorn api:app --reload --app-dir src

Interactive docs (Swagger): http://127.0.0.1:8000/docs
"""

import base64
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
STATIC_DIR = SRC_DIR / "static"
sys.path.insert(0, str(SRC_DIR))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import VECTORSTORE_PATH
from gemini_llm import analyze_medical_image
from rag_chain import create_rag_chain

logger = logging.getLogger("medical_rag.api")

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB
DEFAULT_IMAGE_QUESTION = (
    "Please explain this medical image in simple language."
)

# Objects created once at startup and shared by all requests
state = {}


# -----------------------------------
# STARTUP / SHUTDOWN
# -----------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
    logger.info("Loading RAG chain...")
    state["rag_chain"] = create_rag_chain()
    logger.info("RAG chain ready.")
    yield
    state.clear()


app = FastAPI(
    title="Medical RAG API",
    description=(
        "Multilingual (English / Telugu / Tenglish) medical information "
        "assistant. General information only, not medical advice."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.get("/", include_in_schema=False)
def frontend():
    """Serve the browser-based chat interface."""
    return FileResponse(STATIC_DIR / "index.html")


# Allow browser-based frontends (React etc.) to call the API.
# Restrict allow_origins to your real frontend URL in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------
# SCHEMAS
# -----------------------------------

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        examples=["hypertension lakshanalu enti?"],
    )


class AnswerResponse(BaseModel):
    answer: str


class HealthResponse(BaseModel):
    status: str
    rag_ready: bool


# -----------------------------------
# ENDPOINTS
# -----------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    """Is the server up and is the knowledge base available?"""
    rag_ready = "rag_chain" in state and VECTORSTORE_PATH.is_dir()
    return HealthResponse(status="ok", rag_ready=rag_ready)


@app.post("/ask", response_model=AnswerResponse, tags=["chat"])
def ask(request: AskRequest):
    """Answer a medical question using the PDF knowledge base."""

    question = request.question.strip()

    if len(question) < 2:
        raise HTTPException(status_code=422, detail="Question is too short.")

    rag_chain = state.get("rag_chain")

    if rag_chain is None:
        raise HTTPException(
            status_code=503, detail="RAG system is not ready yet."
        )

    if not VECTORSTORE_PATH.is_dir():
        raise HTTPException(
            status_code=503,
            detail="Medical knowledge base is not built yet.",
        )

    try:
        answer = rag_chain(question)

    except Exception:
        logger.exception("RAG chain failed")
        raise HTTPException(
            status_code=500, detail="Unable to generate an answer."
        )

    return AnswerResponse(answer=answer)


@app.post("/analyze-image", response_model=AnswerResponse, tags=["chat"])
def analyze_image(
    image: UploadFile = File(..., description="PNG or JPEG image"),
    question: str = Form("", description="Optional question about the image"),
):
    """Explain an uploaded medical image (medicine strip, report...)."""

    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Only PNG and JPEG images are supported.",
        )

    image_bytes = image.file.read(MAX_IMAGE_BYTES + 1)

    if not image_bytes:
        raise HTTPException(status_code=400, detail="The image is empty.")

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image is too large (max {MAX_IMAGE_BYTES // (1024 * 1024)} MB).",
        )

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    try:
        answer = analyze_medical_image(
            image_base64,
            question.strip() or DEFAULT_IMAGE_QUESTION,
            image.content_type,
        )

    except Exception:
        logger.exception("Image analysis failed")
        raise HTTPException(
            status_code=500, detail="Unable to analyze the medical image."
        )

    return AnswerResponse(answer=answer)
