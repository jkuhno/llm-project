import requests # type: ignore

from langchain_ollama import ChatOllama # type: ignore
from ollama import Client # type: ignore

class OllamaServer:
    def __init__(self, model: str = "llama3.2", host: str = "http://ollama-server:11434"):
        self.model = model
        self.ollama_host = host

    def pull_model(self):
        url = f"{self.ollama_host}/api/pull"
        payload = {"model": self.model}

        try:
            # Send a POST request
            response = requests.post(url, json=payload)

            # Check the response status
            if response.status_code == 200:
                print("Model loaded successfully!")
                print("Response:", response.json())
            else:
                print("Failed to load model.")
                print("Status code:", response.status_code)
                print("Response:", response.text)

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

    def get_langchain_model(self):
        return ChatOllama(model=self.model, base_url=self.ollama_host)
        
    def get_direct_model(self):
    	return Client(host=self.ollama_host, headers={"Content-Type": "application/json"}
    )