
import os
from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from ingest import get_embedding_model, CHROMA_DIR, COLLECTION_NAME


chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(anonymized_telemetry=False),
)

class Retriever:
    """
    Wraps ChromaDB querying with embedding.
    Single instance is created at startup and reused across requests.
    """

    def __init__(self, top_k: int = 4):
        """
        top_k : number of chunks to return per query.
                4 is a safe default — enough context, not too noisy.
        """
        self.top_k       = top_k
        self.embed_model = get_embedding_model()
        self.collection  = self._load_collection()

    def _load_collection(self):
        """Load the ChromaDB collection. Raises if not yet ingested."""
        try:
            collection = chroma_client.get_collection(COLLECTION_NAME)
            count = collection.count()
            print(f"[Retriever] Collection '{COLLECTION_NAME}' loaded — {count} chunk(s).")
            return collection
        except Exception:
            raise RuntimeError(
                f"ChromaDB collection '{COLLECTION_NAME}' not found.\n"
                "  Run:  python ingest.py   to ingest your documents first."
            )

    def _reload_collection(self):
        """Re-attach to collection (called after a new file is uploaded)."""
        self.collection = self._load_collection()

    def retrieve(self, query: str) -> List[Dict]:
        """
        Embed the query, search ChromaDB, return top-k results.

        Returns a list of dicts:
        [
            {
                "text":   "chunk content ...",
                "source": "filename.pdf",
                "page":   "3",
                "score":  0.87         # cosine similarity (0–1, higher = better)
            },
            ...
        ]
        """
        if not query.strip():
            return []

        query_embedding = self.embed_model.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(self.top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks      = results["documents"][0]   
        metadatas   = results["metadatas"][0]   
        distances   = results["distances"][0]   

        retrieved = []
        for text, meta, dist in zip(chunks, metadatas, distances):
            score = round(1 - dist, 4)          
            retrieved.append({
                "text":   text,
                "source": meta.get("source", "unknown"),
                "page":   meta.get("page", ""),
                "score":  score,
            })

        return retrieved

    def retrieve_as_context(self, query: str) -> str:
        """
        Same as retrieve(), but returns a single formatted string
        ready to be inserted into the LLM prompt.

        Format:
            [Source: file.pdf | Page: 3]
            chunk text here ...

            [Source: file.pdf | Page: 5]
            another chunk ...
        """
        chunks = self.retrieve(query)

        if not chunks:
            return "No relevant context found in the knowledge base."

        parts = []
        for i, chunk in enumerate(chunks, 1):
            page_info = f" | Page: {chunk['page']}" if chunk["page"] else ""
            header    = f"[{i}. Source: {chunk['source']}{page_info}  (score: {chunk['score']})]"
            parts.append(f"{header}\n{chunk['text']}")

        return "\n\n".join(parts)

    def collection_stats(self) -> Dict:
        """Return basic stats about the current collection."""
        count = self.collection.count()
        return {
            "collection": COLLECTION_NAME,
            "total_chunks": count,
            "top_k": self.top_k,
        }

def get_retriever(top_k: int = 4) -> Retriever:
    """
    Returns a module-level Retriever singleton.
    Lazy-init so ingest.py can run without main.py loading this.
    """
    global retriever
    if retriever is None:
        retriever = Retriever(top_k=top_k)
    return retriever

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is this document about?"

    print("=" * 55)
    print("  RAG Chatbot — Retrieval Test")
    print("=" * 55)
    print(f"  Query: {query}\n")

    r = get_retriever(top_k=4)

    print(f"  Collection stats: {r.collection_stats()}\n")
    print("  Top chunks retrieved:")
    print("-" * 55)

    results = r.retrieve(query)
    if not results:
        print("  No results found.")
    else:
        for i, chunk in enumerate(results, 1):
            print(f"\n  [{i}] Score: {chunk['score']}  |  Source: {chunk['source']}  |  Page: {chunk['page']}")
            print(f"      {chunk['text'][:200].strip()} ...")

    print("\n" + "=" * 55)