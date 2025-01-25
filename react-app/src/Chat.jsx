import { useState } from "react";
import "./Chat.css"; // Import the CSS file

function ChatBot() {
  const [messages, setMessages] = useState([
    { sender: "bot", text: "Hi! How are you today?" },
  ]);

  const [userInput, setUserInput] = useState("");

  let BACKEND_ADDRESS = import.meta.env.VITE_APP_BACKEND_ADDRESS;

  const handleSend = async () => {
    if (!userInput.trim()) return;

    // Step 1: Add the user's message to the chat
    setMessages([...messages, { sender: "user", text: userInput }]);

    try {
      // Step 2: Send the user input to the backend via POST
      // Assume the backend is running on http://localhost:8080/generate
      // const response = await fetch("http://localhost:8080/generate", {
      const response = await fetch(BACKEND_ADDRESS, {  
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ input: userInput }), // Send the user input as JSON
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json(); // Assume backend returns JSON with `response` field

      // Step 3: Update the bot's response with the backend result
      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: "bot", text: data.response || "I couldn't process that." },
      ]);
    } catch (error) {
      console.error("Error communicating with backend:", error);
      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: "bot", text: "Sorry, there was an error. Please try again." },
      ]);
    }

    setUserInput(""); // Clear the input after sending
  };

  return (
    <div className="wrapper">
      <div className="container">
        <div className="chatArea">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`messageBubble ${msg.sender === "user" ? "userBubble" : "botBubble"}`}
            >
              {msg.text}
            </div>
          ))}
        </div>
        <div className="inputArea">
          <input
            type="text"
            className="input"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            placeholder="Type your message..."
          />
          <button className="button" onClick={handleSend}>
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatBot;