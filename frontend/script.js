const API_URL = "http://127.0.0.1:8000/chat";  // your FastAPI server

async function sendMessage() {
    const input = document.getElementById("user-input");
    const question = input.value.trim();

    // Don't send empty messages
    if (!question) return;

    // Show user's message in chat
    addMessage(question, "user");
    input.value = "";

    // Show loading indicator
    const loadingEl = addMessage("Thinking...", "loading");

    try {
        // Send question to FastAPI backend
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: question })
        });

        const data = await response.json();

        // Remove loading, show answer
        loadingEl.remove();
        addMessage(data.answer, "bot");

    } catch (error) {
        loadingEl.remove();
        addMessage("Error: Could not reach the server.", "bot");
    }
}

// Helper: adds a message bubble to chat
function addMessage(text, type) {
    const chatBox = document.getElementById("chat-box");
    const div = document.createElement("div");
    div.className = `message ${type}`;
    div.textContent = text;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;  // auto scroll down
    return div;
}