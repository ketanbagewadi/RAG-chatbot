from sentence_transformers import SentenceTransformer
import chromadb


CHROMA_FOLDER = "vectorstore/"
COLLECTION_NAME = "company_docs"
TOP_K = 3                                                # how many chunks to return



#  Load model + Database (ChromaDB) ========================================================

model = SentenceTransformer("all-MiniLM-L6-v2")  
client = chromadb.PersistentClient(path=CHROMA_FOLDER)
collection = client.get_or_create_collection(COLLECTION_NAME)



#  Main Function =================================================================

def retrieve(question: str):
    # Step 1: Convert question to vector
    question_vector = model.encode(question).tolist()

    # Step 2: Search ChromaDB for similar chunks
    results = collection.query(
        query_embeddings=[question_vector],
        n_results=TOP_K
    )

    # Step 3: Return the matching text chunks
    chunks = results["documents"][0]  # list of top 3 texts
    return chunks


# ─── Test it ─────────────────────────────────────────────
if __name__ == "__main__":
    question = "What is the refund policy?"
    chunks = retrieve(question)

    print(f"\nTop {TOP_K} relevant chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ---")
        print(chunk)
        print()