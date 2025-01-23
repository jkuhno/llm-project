from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, trim_messages

import uuid

from ollama_chat_model import OllamaChatModel
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
chat_model = None
graph = None


# Model configs
device = "cuda"
embeddings_model_name = "sentence-transformers/all-mpnet-base-v2"


################################# MODELS INIT ####################################
##################################################################################


@app.on_event("startup")
async def load_models():
    global chat_model, graph #asr_model, fastspeech2, hifi_gan, chat_model, graph    
    # default is "llama 3.2" running on "http://ollama-server:11434"
    model_init = OllamaChatModel()
    model_init.pull_model()
    chat_model = model_init.get_model()

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    dims = 768 # From hf hub model page

    trimmer = trim_messages(
            max_tokens=50,
            strategy="last",
            token_counter=chat_model,
            include_system=True,
            allow_partial=False,
            start_on="human",
    )

    graph_connection = GmapGraph(embeddings, dims, trimmer, chat_model, GMAPS_API_KEY)
    graph = graph_connection.get_graph()

    print("\nModels loaded successfully!")



################################# API ############################################
##################################################################################


@app.post("/generate", response_model=TextResponse)
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

        input_messages = [HumanMessage(query)]
        output = graph.invoke({"messages": input_messages}, config)
        
        # output contains all messages in state
        response = output["messages"][-1]
        output["messages"][-1].pretty_print()
        
        return {"response": response.content}


    except Exception as e:
        print(e)
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred while processing the file: {str(e)}"},
        )

    