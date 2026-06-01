import os
import sys
import hashlib
from pathlib import Path
from typing import List, Dict

import chromadb
from chromadb.config import Settings

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

#first try OpenAI first, if no API key then fall back to Ollama embeddings.
try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_community.embeddings import OllamaEmbeddings
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False



BASE_DIR   = Path(__file__).parent          
DATA_DIR   = BASE_DIR / "data"              
CHROMA_DIR = BASE_DIR / "chroma_db"         

DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)



# ChromaDB client==========================================================================

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(anonymized_telemetry=False),
)
COLLECTION_NAME = "rag_documents"



def get_embedding_model():
    """
    Returns an embedding model.
    Priority:
      1. OpenAI  (if OPENAI_API_KEY env var is set)
      2. Ollama  (local, model: nomic-embed-text)
    Raises RuntimeError if neither is available.
    """
    if OPENAI_AVAILABLE:
        print("[Embeddings] Using OpenAI text-embedding-3-small")
        return OpenAIEmbeddings(model="text-embedding-3-small")

    if OLLAMA_AVAILABLE:
        print("[Embeddings] Using Ollama nomic-embed-text (local)")
        return OllamaEmbeddings(model="nomic-embed-text")

    raise RuntimeError(
        "No embedding model available.\n"
        "  Option 1: Set OPENAI_API_KEY environment variable.\n"
        "  Option 2: Install Ollama and run: ollama pull nomic-embed-text"
    )


# Load Documents

LOADER_MAP = {
    ".pdf":  PyPDFLoader,
    ".txt":  TextLoader,
    ".docx": Docx2txtLoader,
    ".md":   UnstructuredMarkdownLoader,
}

def load_document(file_path: Path) -> List[Document]:
    """Load a single file and return a list of LangChain Documents."""
    suffix = file_path.suffix.lower()
    loader_cls = LOADER_MAP.get(suffix)

    if loader_cls is None:
        print(f"  [Skip] Unsupported file type: {file_path.name}")
        return []

    try:
        loader = loader_cls(str(file_path))
        docs = loader.load()
        # Attach source metadata
        for doc in docs:
            doc.metadata["source"] = file_path.name
        print(f"  [Loaded] {file_path.name}  ({len(docs)} page/section(s))")
        return docs
    except Exception as e:
        print(f"  [Error] Could not load {file_path.name}: {e}")
        return []


def load_all_documents(directory: Path = DATA_DIR) -> List[Document]:
    """Load every supported file from the data directory."""
    all_docs: List[Document] = []
    files = [f for f in directory.iterdir() if f.is_file()]

    if not files:
        print(f"[Warning] No files found in {directory}")
        return []

    print(f"\n[Ingest] Found {len(files)} file(s) in {directory}")
    for file_path in files:
        all_docs.extend(load_document(file_path))

    return all_docs


# chunking

def split_documents(docs: List[Document]) -> List[Document]:
    """
    Split documents into overlapping chunks for better retrieval.
    chunk_size=500 tokens, overlap=50 tokens.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[Chunk]  {len(docs)} document(s) → {len(chunks)} chunk(s)")
    return chunks


# embedding + store in chromaDB

def chunk_id(chunk: Document, index: int) -> str:
    """Generate a stable, unique ID for each chunk."""
    content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]
    source = chunk.metadata.get("source", "unknown")
    return f"{source}_{index}_{content_hash}"


def store_chunks(chunks: List[Document], embed_model, reset: bool = False) -> None:
    """
    Embed chunks and upsert them into ChromaDB.
    Set reset=True to wipe and rebuild the collection from scratch.
    """
    if reset:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
            print(f"[Chroma] Collection '{COLLECTION_NAME}' reset.")
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )

    ids        : List[str]        = []
    documents  : List[str]        = []
    metadatas  : List[Dict]       = []
    embeddings : List[List[float]]= []

    print(f"[Embed]  Embedding {len(chunks)} chunk(s) …")
    for i, chunk in enumerate(chunks):
        text = chunk.page_content.strip()
        if not text:
            continue

        embedding = embed_model.embed_query(text)

        ids.append(chunk_id(chunk, i))
        documents.append(text)
        metadatas.append({
            "source": chunk.metadata.get("source", "unknown"),
            "page":   str(chunk.metadata.get("page", "")),
        })
        embeddings.append(embedding)

        # Progress indicator every 20 chunks
        if (i + 1) % 20 == 0:
            print(f"         … {i + 1}/{len(chunks)} embedded")

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    print(f"[Chroma] Stored {len(ids)} chunk(s) in collection '{COLLECTION_NAME}'.")


def ingest_file(file_path: Path) -> Dict:
    """
    Ingest a single file into ChromaDB.
    Called from the /upload endpoint in main.py.
    Returns a status dict.
    """
    embed_model = get_embedding_model()
    docs   = load_document(file_path)
    if not docs:
        return {"status": "error", "message": f"Could not load {file_path.name}"}

    chunks = split_documents(docs)
    store_chunks(chunks, embed_model, reset=False)   # upsert, don't reset
    return {
        "status":  "success",
        "file":    file_path.name,
        "chunks":  len(chunks),
    }


def ingest_all(reset: bool = True) -> Dict:
    """
    Ingest ALL files in the data/ directory.
    Called from the CLI (see __main__ below).
    reset=True rebuilds the collection from scratch.
    """
    embed_model = get_embedding_model()
    docs   = load_all_documents()
    if not docs:
        return {"status": "error", "message": "No documents found in data/"}

    chunks = split_documents(docs)
    store_chunks(chunks, embed_model, reset=reset)
    return {
        "status":  "success",
        "chunks":  len(chunks),
    }


# main

if __name__ == "__main__":
    print("=" * 55)
    print("  RAG Chatbot — Document Ingestion")
    print("=" * 55)

    reset_flag = "--reset" in sys.argv   # python ingest.py --reset
    result = ingest_all(reset=reset_flag)

    print("\n" + "=" * 55)
    if result["status"] == "success":
        print(f"  Done! {result['chunks']} chunks ready for retrieval.")
    else:
        print(f"  Error: {result['message']}")
    print("=" * 55)