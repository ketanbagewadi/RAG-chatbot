"""
main.py — RAG Chatbot FastAPI Backend
=======================================
Endpoints:
  GET  /health        — status check, shows LLM mode + collection stats
  POST /chat          — SSE streaming chat with RAG
  POST /upload        — upload a file, ingest it into ChromaDB
  GET  /documents     — list ingested documents
  DELETE /documents   — clear the entire collection

Run:
  uvicorn main:app --reload --port 8000
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, AsyncGenerator

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ingest import ingest_file, ingest_all, DATA_DIR, CHROMA_DIR, chroma_client, COLLECTION_NAME
from retriever import get_retriever
from llm import stream_response, get_llm_info


# ══════════════════════════════════════════════════════════════════════════════
# APP SETUP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="RAG Chatbot API",
    description="Retrieval-Augmented Generation chatbot backend",
    version="1.0.0",
)

# Allow the HTML/JS frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy-load retriever (only after at least one ingest) ──────────────────────
_retriever = None

def get_active_retriever():
    """Return the singleton retriever, initializing if needed."""
    global _retriever
    if _retriever is None:
        _retriever = get_retriever(top_k=4)
    return _retriever


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class ChatMessage(BaseModel):
    role: str       # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    question: str
    history: List[ChatMessage] = []   # previous turns for multi-turn chat


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

# ── GET /health ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """
    Returns the current status of the API:
    - LLM mode (openai / ollama) and model name
    - Number of chunks in ChromaDB
    """
    llm_info = get_llm_info()

    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
        chunk_count = collection.count()
        db_status = "ready"
    except Exception:
        chunk_count = 0
        db_status = "empty — run ingest first"

    return {
        "status":      "ok",
        "llm":         llm_info,
        "vector_db":   {"status": db_status, "chunks": chunk_count},
    }


# ── POST /chat  (SSE streaming) ────────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Main chat endpoint. Returns a Server-Sent Events (SSE) stream.

    Flow:
      1. Retrieve top-k relevant chunks from ChromaDB
      2. Build prompt (context + history + question)
      3. Stream LLM response token by token

    Frontend reads the stream like:
      const es = new EventSource(...)   — or via fetch with ReadableStream
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # ── Step 1: Retrieve context ───────────────────────────────────────────────
    try:
        retriever = get_active_retriever()
        context   = retriever.retrieve_as_context(req.question)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # ── Step 2: Convert history pydantic → plain dicts ─────────────────────────
    history = [{"role": m.role, "content": m.content} for m in req.history]

    # ── Step 3: Stream LLM response ────────────────────────────────────────────
    async def token_generator() -> AsyncGenerator[str, None]:
        try:
            # stream_response is a sync generator — wrap for async SSE
            for token in stream_response(context, req.question, history):
                # SSE format: each event is "data: <content>\n\n"
                yield f"data: {token}\n\n"
            # Signal stream end
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering if behind proxy
        },
    )


# ── POST /upload ───────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Upload a document (PDF, TXT, DOCX, MD) and ingest it into ChromaDB.
    The file is saved to backend/data/ then processed by ingest.py.
    """
    allowed_extensions = {".pdf", ".txt", ".docx", ".md"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed_extensions}",
        )

    # Save the uploaded file to data/
    save_path = DATA_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Ingest into ChromaDB
    result = ingest_file(save_path)

    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])

    # Reload retriever so new chunks are immediately searchable
    global _retriever
    if _retriever is not None:
        _retriever._reload_collection()

    return {
        "message": f"File '{file.filename}' ingested successfully.",
        "chunks":  result["chunks"],
    }


# ── GET /documents ─────────────────────────────────────────────────────────────

@app.get("/documents")
def list_documents():
    """
    List all files that have been uploaded to the data/ directory.
    """
    files = [f.name for f in DATA_DIR.iterdir() if f.is_file()]
    return {"documents": files, "count": len(files)}


# ── DELETE /documents ──────────────────────────────────────────────────────────

@app.delete("/documents")
def clear_documents():
    """
    Delete all ingested documents and wipe ChromaDB.
    USE WITH CAUTION — this is irreversible.
    """
    global _retriever

    # Delete all files in data/
    deleted = []
    for f in DATA_DIR.iterdir():
        if f.is_file():
            f.unlink()
            deleted.append(f.name)

    # Wipe ChromaDB collection
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    # Reset retriever
    _retriever = None

    return {
        "message":  "All documents and vector DB cleared.",
        "deleted":  deleted,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)