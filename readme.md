rag-chatbot/
├── backend/
│   ├── main.py              # FastAPI app, chat endpoint
│   ├── ingest.py            # Load docs → chunk → embed → store in Chroma
│   ├── retriever.py         # Query Chroma, return top-k chunks
│   ├── llm.py               # Call LLM with context + question
│   ├── requirements.txt
│   └── data/                # Drop your PDFs/docs here
│
├── frontend/
│   ├── index.html
│   
└── README.md


====================================================================================


ingest.py — 
RAG Chatbot Document Ingestion Pipeline

Loads documents from the /data folder, splits them into chunks,converts them into numbers
embeds them, and stores them in ChromaDB.

Supported file types: PDF, TXT, DOCX, MD