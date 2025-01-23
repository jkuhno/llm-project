import { useState } from "react";
import "./Chat.css"; // Import the CSS file

function ChatBot() {
  const [messages, setMessages] = useState([
    { sender: "bot", text: "Hi! How are you today?" },
  ]);

  const [userInput, setUserInput] = useState("");

  function handleSend() {
    if (!userInput.trim()) return;

    setMessages([...messages, { sender: "user", text: userInput }]);

    setTimeout(() => {
      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: "bot", text: "I received: " + userInput },
      ]);
    }, 1000);

    setUserInput("");
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