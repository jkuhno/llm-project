from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List
import os
import json

import numpy as np
import torch
from datasets import load_dataset

import torchaudio
import librosa

from transformers import pipeline
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, AutoModelForCausalLM, AutoTokenizer



### API server housing AI models
### Serving model outputs via FastAPI
### Copyright (c) Jani Kuhno

class TextRequest(BaseModel):
    text: str

class TextResponse(BaseModel):
    response: str

class AudioResponse(BaseModel):
    samplerate: int
    audio: List[float]  # Audio waveform as a list of floats


app = FastAPI()

device = "cuda:0"  # Change to "cpu" if not using CUDA
dtype = torch.bfloat16  # use float16 or float32 if bfloat is not supported
whisper_dtype = torch.float16 



################################# MODELS #########################################
##################################################################################
# Loading the models in global scope at starup, without using FastAPI on_startup
# Loading the models from cahce, after compose down load from HF hub
# Loading from internet seems to be faster than loading from HDD

print("initializing zephyr model")

gen_model_name = "stabilityai/stablelm-zephyr-3b"

gen_tokenizer = AutoTokenizer.from_pretrained(
    gen_model_name
)

gen_model = AutoModelForCausalLM.from_pretrained(
    gen_model_name,
    torch_dtype=dtype,
    device_map="auto",
    #attn_implementation="sdpa"
)


print("initializing whisper model")

whisper = pipeline(
    "automatic-speech-recognition", 
    "distil-whisper/distil-medium.en", 
    torch_dtype=whisper_dtype, 
    device=device, 
    return_timestamps=True,
)


print("initializing text-to-speech model")

# SpeexhT5forTexttoSpeech and T5HifiGan do not support device_map auto,
# carefully move the models and all input and embedding tensors to gpu manually
processor = SpeechT5Processor.from_pretrained(
    "microsoft/speecht5_tts",
    torch_dtype=dtype,
    device_map="auto",
)
speech_model = SpeechT5ForTextToSpeech.from_pretrained(
    "microsoft/speecht5_tts",
    torch_dtype=dtype,
).to(device)
vocoder = SpeechT5HifiGan.from_pretrained(
    "microsoft/speecht5_hifigan",
    torch_dtype=dtype,
).to(device)

dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = dataset[0]["xvector"]  # Use the first speaker embedding
# Convert to tensor and add batch dimension
# Then send to GPU and convert to bfloat so all tensors are of same type
speaker_embedding = torch.tensor(speaker_embedding).unsqueeze(0).to(device).to(dtype)  



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
        
        data, samplerate = librosa.load(save_path, sr=16000)
        
        transcription = whisper(data)


        ### Generative model ###
        ########################
        messages = [
        {"role": "system", "content": "You are a friendly assistant who always responds in ten words or less."},
        {"role": "user", "content": transcription["text"]},
        ]

        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        inputs = gen_tokenizer(prompt, return_tensors="pt").to(device)

        outputs = gen_model.generate(
                **inputs,
                max_new_tokens=64,
                pad_token_id=gen_tokenizer.eos_token_id  # Prevent errors in some models
        )

        generated_text = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the content of the last assistant response
        answer = generated_text.split("\n")[-1].strip()

        return {"response": answer}

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
    text = request.text

    # Prepare input text
    inputs = processor(text=text, return_tensors="pt")

    # Move the inputs to gpu manually
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate speech
    with torch.no_grad():
        speech = speech_model.generate_speech(inputs["input_ids"], speaker_embeddings=speaker_embedding, vocoder=vocoder)

    # Return the PyTorch tensor from GPU to CPU mem, convert the audio tensor to NumPy 
    # and get the sampling rate
    # Conversion from bfloat to float is needed because numpy doesn't want bfloats
    audio_waveform = speech.to(torch.float32).cpu().numpy()
    sampling_rate = 16000  # SpeechT5 outputs audio at 16kHz

    #pydantic AudioResponse
    return {"samplerate": sampling_rate, "audio": audio_waveform.tolist()}
