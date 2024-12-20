from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List
import os
from tqdm import tqdm
import logging

import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from speechbrain.inference.TTS import FastSpeech2
from speechbrain.inference.vocoders import HIFIGAN

from faster_whisper import WhisperModel


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
lang_app = None

# Model configs
device = "cuda"
config = {"configurable": {"thread_id": "abc123"}}

################################# MODELS #########################################
##################################################################################

# Startup: Load models into memory once
# Use cuda device everywhere it is possible
# Disk so slow that no need to save locally
@app.on_event("startup")
async def load_models():
    global asr_model, fastspeech2, hifi_gan, chat_model, lang_app

    class SuppressInfoLogs:
        """Context manager to suppress only INFO-level logs."""
        def __enter__(self):
            self.original_level = logging.getLogger().level
            logging.getLogger().setLevel(logging.WARNING)

        def __exit__(self, exc_type, exc_value, traceback):
            logging.getLogger().setLevel(self.original_level)

    
    with tqdm(total=4, desc="Loading Models", ncols=80) as pbar:
        
        # 1. ASR: Load Faster-Whisper model (streaming speech-to-text)
        # Experiment with larger models if inaccurate, this is fast enough
        with SuppressInfoLogs():
            asr_model = WhisperModel("tiny", device=device, compute_type="float16")
        pbar.update(1)
        pbar.set_postfix_str("ASR model loaded")

    
        # 2. Llama 3.2 text generation
        with SuppressInfoLogs():
            model_name = "meta-llama/Llama-3.2-3B-Instruct"
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

            llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                quantization_config=quantization_config
            )

            # To support langchain, a pipeline is a must (?)
            # With 8bit quantization, a pipeline is reported to cause slowdown. Needs to be optimized
            pipe = pipeline(
                "text-generation",
                model=llm_model,
                tokenizer=llm_tokenizer,
                max_new_tokens=50,
                return_full_text=False
            )
            hf_pipe = HuggingFacePipeline(pipeline=pipe)
            chat_model = ChatHuggingFace(llm=hf_pipe)
        pbar.update(1)
        pbar.set_postfix_str("Chat model loaded")


        # Load TTS models (FastSpeech2 and HiFi-GAN)
        with SuppressInfoLogs():
            fastspeech2 = FastSpeech2.from_hparams(
                source="speechbrain/tts-fastspeech2-ljspeech",
                run_opts={"device": device},
            )
        pbar.update(1)
        pbar.set_postfix_str("FastSpeech2 model loaded")

        with SuppressInfoLogs():
            hifi_gan = HIFIGAN.from_hparams(
                source="speechbrain/tts-hifigan-ljspeech",
                run_opts={"device": device},
            )
        pbar.update(1)
        pbar.set_postfix_str("HiFi-GAN model loaded")


    print("\nModels loaded successfully!")
    print(f"Fast tokenizer enabled: {llm_tokenizer.is_fast}")

    workflow = StateGraph(state_schema=MessagesState)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                ("You are a helpful assistant, who always greets the user with the word 'sir'. "
                 "Answer concisely to any request or question provided by the user. Use fifteen words or less. "
                 "If answering in numbers, use written form. For example: answer 'number ten' and not 'number 10'"
                ),
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    trimmer = trim_messages(
        max_tokens=200,
        strategy="last",
        token_counter=chat_model,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )

    # Define the function that calls the model
    def call_model(state: MessagesState):
        trimmed_messages = trimmer.invoke(state["messages"])
        prompt = prompt_template.invoke(trimmed_messages)
        response = chat_model.invoke(prompt)
        return {"messages": response}

    # Define the (single) node in the graph
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    memory = MemorySaver()
    lang_app = workflow.compile(checkpointer=memory)



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

        input_messages = [HumanMessage(query)]
        output = lang_app.invoke({"messages": input_messages}, config)
        
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