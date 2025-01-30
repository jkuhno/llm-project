import React, { useState, useEffect, useRef } from "react";
import { fetchEventSource } from "@microsoft/fetch-event-source";
import "./Chat.css"; // Import the CSS file

const Chat = () => {
  const [messages, setMessages] = useState([
    { sender: "bot", text: "Hi! How are you today?" },
  ]);
  const [userInput, setUserInput] = useState("");
  const [loading, setLoading] = useState(false);

  const chatAreaRef = useRef(null); // Ref for the chat area
  const inputRef = useRef(null); // Ref for the input field

  let BACKEND_ADDRESS = import.meta.env.VITE_APP_BACKEND_ADDRESS;

  useEffect(() => {
    if (chatAreaRef.current) {
      chatAreaRef.current.scrollTop = chatAreaRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {

    if (!userInput.trim()) return; // Prevent empty messages
    setUserInput(""); // Clear the input field

    setLoading(true);
    setMessages((prev) => [
      ...prev,
      { sender: "user", text: userInput },
      { sender: "bot", text: "" }, // Placeholder bot message
    ]);

    let botText = ""; // Store the streamed text

    await fetchEventSource(BACKEND_ADDRESS, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: userInput }),

      async onmessage(event) {
        try {
          //event is in the format "data: {json.dumps({'response': msg.content})}\n\n"
          const parsed = JSON.parse(event.data);

          botText += parsed.response; // Append new content

          // Dynamically update the last message in the array
          setMessages((prev) => {
            const updatedMessages = [...prev];
            updatedMessages[updatedMessages.length - 1] = {
              ...updatedMessages[updatedMessages.length - 1],
              text: botText, // Update the last bot message
            };
            return updatedMessages;
          });
        } catch (err) {
          console.error("Error parsing JSON:", err);
        }
      },

      onerror(err) {
        console.error("Streaming error:", err);
      },
    });

    
    setLoading(false);
    inputRef.current?.focus();
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") {
      e.preventDefault(); // Prevent default behavior (new line in textarea)
      handleSend(); // Trigger handleSend when Enter key is pressed
    }
  };

  return (
    <div className="wrapper">
      <div className="container">
        <div className="chatArea" ref={chatAreaRef}>
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
            onKeyDown={handleKeyPress}
            placeholder="Type your message..."
            ref={inputRef} // Attach ref to input
          />
          <button className="button" onClick={handleSend} disabled={loading}>
            Send
          </button>
        </div>
      </div>
    </div>
  );
};
  

export default Chat;