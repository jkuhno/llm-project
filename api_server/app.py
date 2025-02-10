from fastapi import FastAPI # type: ignore
from fastapi.responses import JSONResponse, StreamingResponse # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from pydantic import BaseModel # type: ignore
import os
import json

from langchain_core.messages import HumanMessage, trim_messages # type: ignore

import uuid

from api_server.gmap_graph import get_graph


### Serving model outputs via FastAPI
### Copyright (c) Jani Kuhno


# Pydantic requests and responses
class TextRequest(BaseModel):
    input: str

class TextResponse(BaseModel):
    response: str


os.environ["LANGCHAIN_TRACING_V2"] = "true"
# GMAPS_API_KEY = os.environ['GMAPS_API_KEY']

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

################################# MODELS INIT ####################################
##################################################################################
@app.on_event("startup")
async def load_models():
    global graph 
    graph = get_graph({})
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
