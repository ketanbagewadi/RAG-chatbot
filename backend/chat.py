import anthropic
import os
from dotenv import load_dotenv
from retriever import retrieve  # our retriever from Step 2

load_dotenv()  # loads ANTHROPIC_API_KEY from .env file



# ─── Claude client ==============================================================

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))



# ─── Main Function ==============================================================

def chat(question: str) -> str:

    # Step 1: Get relevant chunks from ChromaDB
    chunks = retrieve(question)

    # Step 2: Join chunks into one block of context
    context = "\n\n".join(chunks)

    # Step 3: Build the prompt
    # We tell Claude: "here is the company data, answer only from this"
    prompt = f"""You are a helpful company assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}
Answer:"""

    # Step 4: Send to Claude API and get response
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Step 5: Extract and return the text answer
    return message.content[0].text


#  Test it =======================================================================

if __name__ == "__main__":
    question = "What is the refund policy?"
    answer = chat(question)
    print("Answer:", answer)