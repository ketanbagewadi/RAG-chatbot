# RAG Chatbot

A document-based chatbot that actually reads your files and answers from them. Upload a PDF, ask a question, get an answer — with sources. No hallucinations, no random facts, just your documents.

Built with FastAPI, ChromaDB, LangChain, and plain HTML/CSS/JS. Works with OpenAI API or a fully local setup using Ollama (no internet needed, no API costs).

## What it does

You drop your documents in, it reads them, cuts them into chunks, converts them into embeddings (numbers that represent meaning), and stores them in a local vector database. When you ask a question, it finds the most relevant chunks from your documents and sends them to an LLM along with your question. The LLM then answers based only on what it found.

It also handles general conversation — so saying "hi" won't make it say "I don't have enough context for that."

## Project Structure

```bash
rag-chatbot/
├── backend/
│   ├── ingest.py
│   ├── retriever.py
│   ├── llm.py
│   ├── main.py
│   ├── .env.example
│   └── data/
├── frontend/
│   └── index.html
├── requirements.txt
└── README.md
```

## Tech Stack

**Backend:** FastAPI, LangChain, ChromaDB, Sentence Transformers  
**Frontend:** HTML + Tailwind CSS + vanilla JavaScript  
**LLM Support:** Works with Groq, Ollama, OpenAI, or any local model  

## ingest.py

This is the first thing that runs. It goes into the `data/` folder, reads every file it finds (PDF, TXT, DOCX, MD), and splits each one into small chunks of around 500 characters with a small overlap between chunks so nothing gets cut off awkwardly.

Each chunk then gets passed through an embedding model which converts the text into a list of numbers. Those numbers represent the meaning of the text — similar chunks get similar numbers. Everything gets saved into ChromaDB, a local vector database that lives right inside the project folder.

You can also call `ingest_file()` from the API when someone uploads a file through the UI — it handles that too.

## retriever.py

When someone asks a question, this file handles finding the relevant parts of your documents. It takes the question, runs it through the same embedding model used during ingestion, and searches ChromaDB for the chunks whose numbers are closest to the question's numbers. That's the core idea of semantic search — you're not matching keywords, you're matching meaning.

It returns the top 4 most relevant chunks by default, along with the source filename, page number, and a similarity score. There's also a `retrieve_as_context()` method that formats everything into a clean string ready to drop into the LLM prompt.

## llm.py

This is where the actual answer gets generated. It takes the retrieved context from `retriever.py`, the user's question, and any previous conversation history, and builds a prompt. That prompt gets sent to either OpenAI or Ollama depending on what's configured.

Both modes support streaming — tokens come back one at a time and get sent to the frontend as they arrive, giving that real-time typing effect. The model is set to low temperature (`0.3`) so answers stay factual and don't wander off.

If `OPENAI_API_KEY` is set in your environment, it uses OpenAI. If not, it falls back to Ollama automatically.

## main.py

The FastAPI backend that ties everything together. Four endpoints:

`GET /health` — tells you if the backend is running, which LLM mode is active, and how many chunks are in the database  

`POST /chat` — takes a question and conversation history, runs the full RAG pipeline, streams the response back as Server-Sent Events  

`POST /upload` — accepts a file, saves it, runs ingestion, reloads the retriever so the new content is immediately searchable  

`DELETE /documents` — wipes everything, both the files and the vector database  

CORS is open by default so the plain HTML frontend can talk to it without issues.

## frontend/index.html

A single HTML file, no framework, no build step. Just open it in a browser. The left sidebar has the file upload area (drag and drop works too), a list of all uploaded documents, and a button to clear everything. The right side is the chat interface.

The LLM mode badge at the top shows whether you're running on OpenAI or Ollama, pulled live from `/health`. Responses stream in token by token. The input box auto-resizes as you type. Everything is in one file — HTML, CSS, and JS together.

## LLM Modes

## OpenAI (cloud)

- Needs an API key
- Faster, better quality answers
- Costs money per request
- Uses `gpt-4o-mini` by default (change in `.env`)

## Ollama (local)

- Free, runs on your machine
- No data leaves your computer
- Needs Ollama installed and a model pulled
- Uses `llama3` by default (change in `.env`)

The switch is automatic. If `OPENAI_API_KEY` is set, it uses OpenAI. If not, it uses Ollama. Same goes for embeddings — OpenAI embeddings vs `nomic-embed-text` locally.

## How to run whole project

# 1. Clone and set up the folder

```bash
git clone
cd rag-chatbot
```

# 2. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# 3. Configure your API key in `.env`

Open `.env` and either set your OpenAI key:

```env
OPENAI_API_KEY=sk-your-key-here
```

Or leave it blank and set up Ollama:

```bash
Install Ollama from https://ollama.com

ollama pull llama3
ollama pull nomic-embed-text
ollama serve
```

# 4. Add your documents in data file

Drop any PDF, TXT, DOCX, or MD files into:

```bash
PATH: rag-chatbot/backend/data/
```

# 5. Ingest the documents

```bash
python3 backend/ingest.py
```

You'll see it load your files, split them into chunks, embed them, and store them. At the end it tells you how many chunks were saved.

To wipe and rebuild from scratch:

```bash
python3 backend/ingest.py --reset
```

# 6. Start the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

Leave this running. You'll see Uvicorn running on:

```bash
http://0.0.0.0:8000
```

# 7. Open the frontend

Just open the file directly in your browser:

**Mac**

```bash
open frontend/index.html
```

**Linux**

```bash
xdg-open frontend/index.html
```

Or just double-click the file in your file manager.

The green dot in the top right means the backend is connected and ready.

## Notes

- The `chroma_db/` folder gets created automatically inside `backend/` after the first ingest. This is your local vector database — don't delete it unless you want to re-ingest everything.
- You can upload more files through the UI at any time without restarting the server.
- Conversation history is kept in the browser tab. Refreshing the page starts a fresh conversation, but the documents stay in the database.
- The `.env` file is gitignored by default. Never commit your API key.

#  Use the Docker
#  Run with Docker

# 1. Pull the images

docker pull ketanbagewadi/rag-chatbot-backend:latest
docker pull ketanbagewadi/rag-chatbot-frontend:latest

# 2. Create a docker-compose.yml file

(copy the docker-compose.yml from this repo)

# 3. Add your API key

Edit a file `backend/.env` and add:
OPENAI_API_KEY=sk-your-key-here

# 4. Run

docker-compose up

# 5. Open browser

http://localhost:8000
