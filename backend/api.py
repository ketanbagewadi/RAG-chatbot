from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chat import chat  # our chat function from Step 3


#  App setup ===============================================================
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production, replace * with your domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request model ───────────────────────────────────────
# Defines what the frontend must send: { "question": "..." }
class QuestionRequest(BaseModel):
    question: str

# ─── Routes ─────────────────────────────────────────────

# Health check — just to confirm server is running
@app.get("/")
def root():
    return {"status": "RAG server is running"}

# Main chat endpoint
@app.post("/chat")
def ask(request: QuestionRequest):
    answer = chat(request.question)  # calls chat.py
    return {"answer": answer}