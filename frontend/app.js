document.addEventListener("DOMContentLoaded", () => {
    const messageInput = document.getElementById("input-box");
    const sendButton = document.getElementById("send-button");
    const pauseButton = document.getElementById("pause-button");
    const messagesContainer = document.getElementById("messages-container");
    const acknowledgeButton = document.getElementById("acknowledge-button");
    const ackContainer = document.getElementById("ack-container");
    const chatContainer = document.getElementById("chat-container");
    const spinner = document.getElementById("spinner");

    let conversation = { conversation: [] }; // Local conversation state
    let isLoading = false;
    let controller = null; // To control the fetch request
    const host = "http://localhost:8000"; 
    // local host

    const handleInputChange = (event) => {
        messageInput.value = event.target.value;
    };

    const handleNewSession = () => {
        conversation = { conversation: [] }; // Reset the conversation
        renderMessages();
    };

    const handleSubmit = async () => {
        if (messageInput.value.trim() === '') {
            alert('Please enter a message before sending.'); // Alert the user
            return; // Do nothing if the message is empty
        }

        setLoading(true);

        const newConversation = [
            ...conversation.conversation,
            { role: "user", content: messageInput.value },
        ];

        // Create a new AbortController for each request
        controller = new AbortController();
        const signal = controller.signal;

        try {
            const response = await fetch(`${host}/inference/`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ conversation: newConversation }),
                signal // Pass the signal to the fetch request
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            conversation = data; // Update the local conversation state
            messageInput.value = "";
            setLoading(false);
            renderMessages();
        } catch (error) {
            if (error.name === 'AbortError') {
                console.log('Fetch aborted');
            } else {
                console.error('Fetch error:', error);
            }
            setLoading(false);
            console.log("Connection not made");
        }
    };

    const setLoading = (state) => {
        isLoading = state;
        messageInput.disabled = isLoading;
        sendButton.disabled = isLoading;
        pauseButton.disabled = !isLoading; // Enable pause button only when loading
        spinner.style.display = isLoading ? "flex" : "none"; // Control the spinner visibility
    };

    const renderMessages = () => {
        messagesContainer.innerHTML = "";
        conversation.conversation
            .filter(message => message.role !== "system")
            .forEach(message => {
                const messageDiv = document.createElement("div");
                messageDiv.className = `${message.role}-message`;
                messageDiv.innerHTML = `<span>${message.content}</span>`;
                messagesContainer.appendChild(messageDiv);
            });
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    };

    messageInput.addEventListener("input", handleInputChange);
    sendButton.addEventListener("click", handleSubmit);
    pauseButton.addEventListener("click", () => {
        if (controller) {
            controller.abort(); // Abort the ongoing fetch request
            setLoading(false); // Reset the loading state
        }
    });
    messageInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            event.preventDefault();
            handleSubmit();
        }
    });
    acknowledgeButton.addEventListener("click", () => {
        ackContainer.classList.add("hidden");
        chatContainer.classList.remove("hidden");
    });
});
