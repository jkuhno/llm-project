from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List
import os

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from faster_whisper import WhisperModel

from speechbrain.inference.TTS import FastSpeech2
from speechbrain.inference.vocoders import HIFIGAN

### API server housing AI models
### Serving model outputs via FastAPI
### Copyright (c) Jani Kuhno

os.environ["SUNO_USE_SMALL_MODELS"] = "True"

class TextRequest(BaseModel):
    text: str

class TextResponse(BaseModel):
    response: str

class AudioResponse(BaseModel):
    samplerate: int
    audio: List[float]  # Audio waveform as a list of floats


# FastAPI app
app = FastAPI()

# Model holders
asr_model = None
tokenizer = None
llm_model = None
tts_model = None
fastspeech2 = None
hifi_gan = None

#Model variables
device = "cuda"


# Startup: Load models into memory once
# Use cuda device everywhere it is possible
@app.on_event("startup")
async def load_models():
    global asr_model, tokenizer, llm_model, fastspeech2, hifi_gan

    # 1. ASR: Load Faster-Whisper model (streaming speech-to-text)
    # Experiment with larger models if inaccurate, this is fast enough
    asr_model = WhisperModel("tiny", device=device, compute_type="float16")

    # 2. LLM: Load a small text generation model (distilgpt2 in this case)
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=0)
    llm_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # 3. fastspeech and HIFIGAN
    # Disk so slow that no need to save locally
    fastspeech2 = FastSpeech2.from_hparams(
        source="speechbrain/tts-fastspeech2-ljspeech",
        run_opts={"device": device},
    )
    hifi_gan = HIFIGAN.from_hparams(
        source="speechbrain/tts-hifigan-ljspeech",
        run_opts={"device": device}, 
        )
    print("Models loaded successfully!")



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
            recognized_text = "Error in ASR process [nervous laughter]"


        ### Generative model ###
        ########################
        inputs = tokenizer(recognized_text, return_tensors="pt").to(device)
        output_tokens = llm_model.generate(**inputs, max_new_tokens=15, do_sample=True)
        generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        print(f"Generated Text: {generated_text}")

        return {"response": generated_text}

    except Exception as e:
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