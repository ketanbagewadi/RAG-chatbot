from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import os

DATA_FOLDER = "data/"           
CHROMA_FOLDER = "vectorstore/"  
COLLECTION_NAME = "company_docs"
CHUNK_SIZE = 500                
CHUNK_OVERLAP = 50              

#  Step 1: Load PDFs ,this pdf reader for normal text ================================
# def load_documents():
#     loader = PyPDFDirectoryLoader(DATA_FOLDER)  # reads all PDFs in folder
#     docs = loader.load()
#     print(f"Loaded {len(docs)} pages from PDFs")
#     return docs

from unstructured.partition.pdf import partition_pdf
import glob

def load_documents():                                   #this pdf reads images too
    all_texts = []
    for pdf_path in glob.glob(f"{DATA_FOLDER}/*.pdf"):
        elements = partition_pdf(pdf_path, strategy="hi_res")
        for el in elements:
            all_texts.append(str(el))
    print(f"Loaded {len(all_texts)} elements from PDFs")
    return all_texts



#  Step 2: Split into chunks ==========================================
# def split_documents(docs):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP
#     )
#     chunks = splitter.split_documents(docs)
#     print(f"Split into {len(chunks)} chunks")
#     return chunks


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.create_documents(docs)  # ← changed from split_documents
    return chunks



#  Step 3: Embed + Save to ChromaDB =======================================
def embed_and_store(chunks):
    # Load embedding model (runs locally, no API key needed)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Connect to ChromaDB (creates vectorstore/ folder automatically)
    client = chromadb.PersistentClient(path=CHROMA_FOLDER)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    texts = [chunk.page_content for chunk in chunks]
    ids   = [str(i) for i in range(len(chunks))]

    # Convert text → vectors
    embeddings = model.encode(texts).tolist()

    # Save to ChromaDB
    collection.add(documents=texts, embeddings=embeddings, ids=ids)
    print(f"Stored {len(chunks)} chunks in ChromaDB")



#  Main ========================================================================
if __name__ == "__main__":
    docs   = load_documents()
    chunks = split_documents(docs)
    embed_and_store(chunks)
    print("✅ Done! Documents are ready.")