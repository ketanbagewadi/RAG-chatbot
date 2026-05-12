# RAG-chatbot
Personal Knowledge Chatbot


Architecture Flow

Upload PDF
↓
Text Extraction
↓
Chunking
↓
Embedding Generation
↓
Store in Vector DB
↓
User asks question
↓
Similarity Search
↓
LLM generates answer



rag-chatbot/
│
├── data/                  # Put company PDFs/Docs here
│
├── vectorstore/           # ChromaDB saves here (auto-created)
│
├── backend/
│   ├── ingest.py          # Read docs → chunk → embed → store in ChromaDB
│   ├── retriever.py       # Search ChromaDB for relevant chunks
│   ├── chat.py            # Send chunks + question to Claude API
│   └── api.py             # FastAPI server (main entry point)
│
├── frontend/
│   ├── index.html         # Chat UI
│   ├── style.css          # Styling
│   └── script.js          # Sends questions to FastAPI, shows answers
│
├── .env                   # API keys (Claude)
└── requirements.txt       # All Python libraries


ingest.py → get docs into vector DB
retriever.py → search working
chat.py → Claude answering correctly
api.py → wrap in API
frontend/ → plug in the UI


pip install fastapi chromadb langchain anthropic sentence-transformers uvicorn langchain-community python-dotenv pypdf
pip install unstructured[pdf] tesseract

=================================================================================================================

ingest.py: This file does 3 things:

            Read PDFs from the data/ folder
            Split them into small chunks
            Embed and save them into ChromaDB

=================================================================================================================

backend/retriever.py: 

What it does: Takes the user's question → converts it to a vector → searches ChromaDB → returns the top 3 most relevant chunks.

===================================================================================================================

backend/chat.py:

retriever finds the relevant chunks → we pass those chunks + the user's question to Claude → Claude reads them and gives a grounded answer.

=================================================================================================================

backend/api.py:

It creates a web server with one endpoint /chat. The frontend sends a question to it → it calls chat.py → returns the answer back to frontend.

=================================================================================================================