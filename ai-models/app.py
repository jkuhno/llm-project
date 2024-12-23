from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List
import os
from tqdm import tqdm
import logging

import torch
from huggingface_hub import login
from transformers import AutoProcessor, BitsAndBytesConfig
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, trim_messages

import uuid

from speechbrain.inference.TTS import FastSpeech2
from speechbrain.inference.vocoders import HIFIGAN

from faster_whisper import WhisperModel

from chat_model import ChatModel
from sync_graph import SyncGraph


### API server housing AI models
### Serving model outputs via FastAPI
### Copyright (c) Jani Kuhno


# Pydantic requests and responses
class TextRequest(BaseModel):
    text: str

class TextResponse(BaseModel):
    response: str

class AudioResponse(BaseModel):
    samplerate: int
    audio: List[float]  # Audio waveform as a list of floats


# Set langsmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
with open('/run/secrets/langsmith_token') as token:
    if token:
        os.environ["LANGCHAIN_API_KEY"] = token.read().strip()


# Authenticate with the access token for gated llama models
with open('/run/secrets/hf_token') as token:
    hf_token = token.read().strip()
login(hf_token)


# FastAPI app
app = FastAPI()


# Model global holders
asr_model = None
fastspeech2 = None
hifi_gan = None
chat_model = None
graph = None


# Model configs
device = "cuda"
chat_model_name = "meta-llama/Llama-3.2-3B-Instruct"
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
embeddings_model_name = "sentence-transformers/all-mpnet-base-v2"



################################# MODELS #########################################
##################################################################################

# Startup: Load models into memory once
# Use cuda device everywhere it is possible
# Disk so slow that no need to save locally

@app.on_event("startup")
async def load_models():
    global asr_model, fastspeech2, hifi_gan, chat_model, graph

    class SuppressInfoLogs:
        def __enter__(self):
            self.original_level = logging.getLogger().level
            logging.getLogger().setLevel(logging.WARNING)

        def __exit__(self, exc_type, exc_value, traceback):
            logging.getLogger().setLevel(self.original_level)

    
    with tqdm(total=4, desc="Loading Models", ncols=80) as pbar:
        
        ######################### ASR ######################
        # Load Faster-Whisper model (streaming speech-to-text)
        # Experiment with larger models if inaccurate, this is fast enough
        with SuppressInfoLogs():
            asr_model = WhisperModel("tiny", device=device, compute_type="float16")
        pbar.update(1)
        pbar.set_postfix_str("ASR model loaded")

    
        ###################### CHAT MODEL ###################
        with SuppressInfoLogs():
            model_conf = ChatModel(chat_model_name, quantization_config)
            chat_model = model_conf.get_model()
        pbar.update(1)
        pbar.set_postfix_str("Chat model loaded")


        ################# TTS (FASTSPEECH2) ##################
        with SuppressInfoLogs():
            fastspeech2 = FastSpeech2.from_hparams(
                source="speechbrain/tts-fastspeech2-ljspeech",
                run_opts={"device": device},
            )
        pbar.update(1)
        pbar.set_postfix_str("FastSpeech2 model loaded")


        ##################### TTS (HIFI GAN) ##################
        with SuppressInfoLogs():
            hifi_gan = HIFIGAN.from_hparams(
                source="speechbrain/tts-hifigan-ljspeech",
                run_opts={"device": device},
            )
        pbar.update(1)
        pbar.set_postfix_str("HiFi-GAN model loaded")


    print("\nModels loaded successfully!")


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

    graph_connection = SyncGraph(embeddings, dims, trimmer, chat_model)
    graph = graph_connection.get_graph()



################################# API ############################################
##################################################################################


"""
    Convert input audio file to text and generate an AI response
    
    Args:
        File: The audio file to convert to text and generate response to

    Returns:
        Dict{response: str}: AI generated response to the input audio


    """
@app.post("/generate", response_model=TextResponse)
async def generate_answer(file: UploadFile = File(...)):
    print(file.filename)
    try:
        # Check file type
        if "." not in file.filename:
            raise HTTPException(
                status_code=400, detail="Invalid file type."
            )
        
        # Save the file to disk
        save_path = "received_audio.wav"
        with open(save_path, "wb") as f:
            content = await file.read()  # Read the file asynchronously
            f.write(content)  # Write to the file

        ### Speech recognition ###
        ##########################
        
        segments, _ = asr_model.transcribe(save_path, language="en")
        recognized_text = " ".join(segment.text for segment in segments)
        
        print(f"ASR Output: {recognized_text}")

        if not recognized_text:
            recognized_text = "Error in ASR process, did not produce text"


        ### Generative model ###
        ########################
        
        query = recognized_text

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
        print(f"Error in ASR or generation: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred while processing the file: {str(e)}"},
        )

    



"""
    Convert input text to speech using Hugging Face Transformers TTS models.
    
    Args:
        text (str): The input text to convert to audio.

    Returns:
        Tuple[int, List]: A tuple containing the sampling rate and the audio waveform.
                                   List of floats needs to be converted to ndarray in UI app.
    """

@app.post("/speech", response_model=AudioResponse)
async def read_response(request: TextRequest):
    try:
        text = request.text
        #text = "testing testing"
        # Prepare input text
        mel_output, durations, pitch, energy = fastspeech2.encode_text(
            [text],
            pace=1.0,        # scale up/down the speed
            pitch_rate=1.0,  # scale up/down the pitch
            energy_rate=1.0, # scale up/down the energy
        )

        waveforms = hifi_gan.decode_batch(mel_output)
        waveforms = waveforms.cpu().numpy().squeeze() 

        # https://speechbrain.readthedocs.io/en/latest/API/speechbrain.inference.vocoders.html#speechbrain.inference.vocoders.HIFIGAN.decode_spectrogram
        sampling_rate = 22050

        #pydantic AudioResponse
        return {"samplerate": sampling_rate, "audio": waveforms.tolist()}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred while processing the file: {str(e)}"},
        )