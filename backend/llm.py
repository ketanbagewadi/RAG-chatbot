"""
llm.py — RAG Chatbot LLM Interface
=====================================
Calls the LLM with retrieved context + user question.
Supports:
  - OpenAI API  (gpt-4o, gpt-3.5-turbo, etc.)  — if OPENAI_API_KEY is set
  - Ollama local models  (llama3, mistral, etc.) — fallback

Both modes support STREAMING responses.
"""

import os
from typing import Generator, List, Dict


# ══════════════════════════════════════════════════════════════════════════════
# MODEL CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# OpenAI
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL      = os.getenv("OPENAI_MODEL", "gpt-4o-mini")   # cheap + fast default

# Ollama (local)
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "llama3")        # change to mistral, phi3, etc.

# Detect which mode to use
USE_OPENAI = bool(OPENAI_API_KEY)


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════

# SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions \
# based on the provided context from a knowledge base.

# Rules:
# - Answer ONLY from the context provided below.
# - If the context does not contain enough information, say:
#   "I don't have enough information in the knowledge base to answer that."
# - Be concise and clear.
# - Cite the source document when relevant (e.g. "According to report.pdf ...").
# - Never make up facts not present in the context.
# """


SYSTEM_PROMPT = """You are a helpful AI assistant with access to a knowledge base.

Rules:
- For greetings or general questions (hi, thanks, how are you, etc.), respond naturally and friendly.
- If the context below is relevant to the question, use it to answer and cite the source.
- If the context is NOT relevant or empty, answer from your general knowledge.
- Only say "I don't have that information" if the question is very specific and truly unanswerable.
- Be concise and clear.
"""

# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(context: str, question: str, history: List[Dict] = None) -> List[Dict]:
    """
    Build the messages list for the LLM.

    Args:
        context  : formatted string from retriever.retrieve_as_context()
        question : current user question
        history  : list of previous turns [{"role": "user/assistant", "content": "..."}]

    Returns:
        messages list ready for OpenAI or Ollama /api/chat
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject retrieved context as a system-level knowledge block
    messages.append({
        "role": "system",
        "content": f"--- KNOWLEDGE BASE CONTEXT ---\n{context}\n--- END CONTEXT ---",
    })

    # Add conversation history (last 6 turns max to stay within context window)
    if history:
        messages.extend(history[-6:])

    # Add the current question
    messages.append({"role": "user", "content": question})

    return messages


# ══════════════════════════════════════════════════════════════════════════════
# OPENAI — STREAMING
# ══════════════════════════════════════════════════════════════════════════════

def stream_openai(messages: List[Dict]) -> Generator[str, None, None]:
    """
    Stream tokens from OpenAI API.
    Yields one text chunk at a time.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    client = OpenAI(api_key=OPENAI_API_KEY)

    print(f"[LLM] OpenAI streaming — model: {OPENAI_MODEL}")

    stream = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        stream=True,
        temperature=0.3,      # low = more factual, less hallucination
        max_tokens=1024,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


# ══════════════════════════════════════════════════════════════════════════════
# OLLAMA — STREAMING
# ══════════════════════════════════════════════════════════════════════════════

def stream_ollama(messages: List[Dict]) -> Generator[str, None, None]:
    """
    Stream tokens from a local Ollama model.
    Yields one text chunk at a time.
    Ollama must be running: `ollama serve`
    """
    try:
        import requests
        import json
    except ImportError:
        raise RuntimeError("requests package not installed. Run: pip install requests")

    url = f"{OLLAMA_BASE_URL}/api/chat"

    payload = {
        "model":    OLLAMA_MODEL,
        "messages": messages,
        "stream":   True,
        "options": {
            "temperature": 0.3,
            "num_predict": 1024,
        },
    }

    print(f"[LLM] Ollama streaming — model: {OLLAMA_MODEL}  url: {url}")

    try:
        response = requests.post(url, json=payload, stream=True, timeout=120)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}.\n"
            "  Make sure Ollama is running:  ollama serve\n"
            f"  And the model is pulled:      ollama pull {OLLAMA_MODEL}"
        )

    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
                if data.get("done"):
                    break
            except json.JSONDecodeError:
                continue


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API  (used by main.py)
# ══════════════════════════════════════════════════════════════════════════════

def stream_response(
    context: str,
    question: str,
    history: List[Dict] = None,
) -> Generator[str, None, None]:
    """
    Main entry point called by main.py's /chat endpoint.

    Automatically picks OpenAI or Ollama based on env vars.
    Yields text tokens as a streaming generator.

    Usage in FastAPI (SSE):
        for token in stream_response(context, question, history):
            yield f"data: {token}\\n\\n"
    """
    messages = build_prompt(context, question, history)

    if USE_OPENAI:
        yield from stream_openai(messages)
    else:
        yield from stream_ollama(messages)


def get_llm_info() -> Dict:
    """Return which LLM mode is active — shown in /health endpoint."""
    if USE_OPENAI:
        return {"mode": "openai", "model": OPENAI_MODEL}
    else:
        return {"mode": "ollama", "model": OLLAMA_MODEL, "url": OLLAMA_BASE_URL}


# ══════════════════════════════════════════════════════════════════════════════
# CLI  — test LLM directly:  python llm.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  RAG Chatbot — LLM Test")
    print("=" * 55)
    print(f"  Mode: {get_llm_info()}\n")

    # Fake context to test without needing ChromaDB
    test_context = """
[1. Source: sample.pdf | Page: 1]
The return policy allows customers to return items within 30 days of purchase
with a valid receipt. Items must be unused and in original packaging.

[2. Source: sample.pdf | Page: 2]
Refunds are processed within 5-7 business days to the original payment method.
"""

    test_question = "How long do I have to return an item?"

    print(f"  Question: {test_question}\n")
    print("  Answer (streaming):")
    print("-" * 55)

    for token in stream_response(test_context, test_question):
        print(token, end="", flush=True)

    print("\n" + "=" * 55)