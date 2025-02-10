from langchain_openai import ChatOpenAI # type: ignore
import os

def get_vllm_chat_model():
    inference_server_url = "http://vllm-server:8000/v1"
    model = os.environ["VLLM_MODEL"]
    return ChatOpenAI(
        model=model,
        openai_api_key="EMPTY",
        openai_api_base=inference_server_url,
        max_tokens=200,
        temperature=0,
)