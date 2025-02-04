from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, trim_messages

import uuid

from ollama_chat_model import OllamaServer
from gmap_graph import GmapGraph


### Serving model outputs via FastAPI
### Copyright (c) Jani Kuhno


# Pydantic requests and responses
class TextRequest(BaseModel):
    input: str

class TextResponse(BaseModel):
    response: str


os.environ["LANGCHAIN_TRACING_V2"] = "true"
GMAPS_API_KEY = os.environ['GMAPS_API_KEY']

# FastAPI app
app = FastAPI()

# Avoid CORS issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, or restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model global holders
graph = None


# Model configs
device = "cuda"
embeddings_model_name = "sentence-transformers/all-mpnet-base-v2"

# Ollama server configs
OLLAMA_HOST = "http://ollama-server:11434"
OLLAMA_MODEL_NAME = "llama3.2"

################################# MODELS INIT ####################################
##################################################################################


@app.on_event("startup")
async def load_models():
    global graph # chat_model, graph, direct_ollama_model
    # default is "llama 3.2" running on "http://ollama-server:11434"
    ollama_server = OllamaServer(model=OLLAMA_MODEL_NAME, host=OLLAMA_HOST)
    ollama_server.pull_model()
    chat_model = ollama_server.get_langchain_model()
    direct_ollama_model = ollama_server.get_direct_model()


    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    dims = 768 # From hf hub model page

    trimmer = trim_messages(
            max_tokens=50,
            strategy="last",
            token_counter=chat_model,
            include_system=True,
            allow_partial=False,
            start_on="human",
    )

    graph_connection = GmapGraph(embeddings=embeddings, 
                                 dims=dims, 
                                 trimmer=trimmer, 
                                 chat_model=chat_model, 
                                 direct_ollama_model=direct_ollama_model,
                                 ollama_pulled_model=OLLAMA_MODEL_NAME,
                                 api_key=GMAPS_API_KEY)
    graph = graph_connection.get_graph()

    print("\nModels loaded successfully!")



################################# API ############################################
##################################################################################


@app.post("/generate")
async def generate_answer(user_input: TextRequest):
    try:
        
        query = user_input.input
        
        # TODO> get the user_id and thread_id from the request @jani
        user_id = "user_1"
        thread_id = "abc123"

        input_messages = [HumanMessage(query)]
        config = {"configurable": {
                                  "thread_id": thread_id, 
                                  "user_id": user_id,
                                  "mem_key": uuid.uuid4()
                                  }, 
        }   

        def generator(input_messages):
            for msg, metadata in graph.stream({"messages": input_messages}, 
                                              config=config, 
                                              stream_mode="messages"):
                if msg.content and metadata["langgraph_node"] == "chat":
                    yield f"data: {json.dumps({'response': msg.content})}\n\n"

        return StreamingResponse(generator(input_messages), media_type="text/event-stream")              
        

    except Exception as e:
        print(e)
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred while processing the file: {str(e)}"},
        )
