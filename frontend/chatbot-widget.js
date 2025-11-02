(function () {
    const orgId = document.currentScript.getAttribute("data-org-id");

    // --- Create chat bubble ---
    const chatButton = document.createElement("div");
    chatButton.innerText = "💬 Chat";
    chatButton.style = `
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: #0078ff;
    color: white;
    padding: 14px 20px;
    border-radius: 25px;
    cursor: pointer;
    font-family: sans-serif;
    font-size: 14px;
    z-index: 9999;
  `;
    document.body.appendChild(chatButton);

    // --- Chat box ---
    const chatBox = document.createElement("div");
    chatBox.style = `
    position: fixed;
    bottom: 80px;
    right: 20px;
    width: 320px;
    height: 420px;
    background: white;
    border-radius: 10px;
    display: none;
    flex-direction: column;
    box-shadow: 0 0 10px rgba(0,0,0,0.15);
    overflow: hidden;
    z-index: 9999;
  `;
    const messages = document.createElement("div");
    messages.style =
        "flex: 1; overflow-y: auto; padding: 10px; font-family: sans-serif; font-size:14px;";
    const input = document.createElement("input");
    input.placeholder = "Ask something...";
    input.style =
        "border: none; border-top: 1px solid #ccc; padding: 12px; width: 100%; box-sizing: border-box;";

    chatBox.appendChild(messages);
    chatBox.appendChild(input);
    document.body.appendChild(chatBox);

    chatButton.onclick = () => {
        chatBox.style.display = chatBox.style.display === "none" ? "flex" : "none";
    };

    // --- Handle message send ---
    input.addEventListener("keydown", async (e) => {
        if (e.key === "Enter" && input.value.trim() !== "") {
            const userMsg = input.value.trim();
            input.value = "";
            append("user", userMsg);

            const formData = new FormData();
            formData.append("orgId", orgId);
            formData.append("message", userMsg);

            try {
                const res = await fetch("http://localhost:5000/chat", {
                    method: "POST",
                    body: formData,
                });
                const data = await res.json();
                append("bot", data.response);
            } catch (err) {
                append(
                    "bot",
                    "Error contacting chatbot server. Please check your connection."
                );
            }
        }
    });

    // --- Helper to append messages ---
    function append(sender, text) {
        const bubble = document.createElement("div");
        bubble.textContent = text;
        bubble.style =
            sender === "user"
                ? "text-align: right; margin: 8px; color: #0078ff;"
                : "text-align: left; margin: 8px; color: #333;";
        messages.appendChild(bubble);
        messages.scrollTop = messages.scrollHeight;
    }
})();