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


class AudioResponse(BaseModel):
    samplerate: int
    audio: List[float]  # Audio waveform as a list of floats


app = FastAPI()


def generate_answer(transcription):

    model_name = "stabilityai/stablelm-zephyr-3b"
    device = "cuda:0"  # Change to "cpu" if not using CUDA
    dtype = torch.bfloat16  # Use float16 or float32 for other devices

    if os.path.exists(f"/usr/src/app/models/{model_name}"):
        print(f"{model_name} saves found")
    
        tokenizer = AutoTokenizer.from_pretrained(f"/usr/src/app/models/{model_name}", use_safetensors=True)
        model = AutoModelForCausalLM.from_pretrained(f"/usr/src/app/models/{model_name}", use_safetensors=True, torch_dtype=dtype).to(device)

    else:
        print(f"No {model_name} saves, downloading and saving models")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)

        tokenizer.save_pretrained(f"/usr/src/app/models/{model_name}", safe_serialization=True)
        model.save_pretrained(f"/usr/src/app/models/{model_name}", safe_serialization=True)

        print(f"{model_name} saved")

    messages = [
        {"role": "system", "content": "You are a friendly assistant who always responds in ten words or less."},
        {"role": "user", "content": transcription},
    ]
    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        pad_token_id=tokenizer.eos_token_id  # Prevent errors in some models
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract the content of the last assistant response
    return generated_text.split("\n")[-1].strip()



@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
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


        whisper = pipeline("automatic-speech-recognition", "distil-whisper/distil-medium.en", torch_dtype=torch.float16, device="cuda:0", return_timestamps=True)
        
        #data, samplerate = sf.read(save_path)
        data, samplerate = librosa.load(save_path, sr=16000)
        

        transcription = whisper(data)
        answer = generate_answer(transcription["text"])
        return answer

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred while processing the file: {str(e)}"},
        )


@app.post("/generate", response_model=AudioResponse)
async def read_response(request: TextRequest):
    """
    Convert input text to speech using Hugging Face Transformers TTS models.
    
    Args:
        text (str): The input text to convert to audio.

    Returns:
        Tuple[int, numpy.ndarray]: A tuple containing the sampling rate and the audio waveform.
                                   This output is compatible with Gradio Audio element.
    """
    text = request.text

    if os.path.exists("/usr/src/app/models/microsoft/speecht5_tts"):
        print("speecht5_tts saves found")

        # Load the processor and model
        processor = SpeechT5Processor.from_pretrained("/usr/src/app/models/microsoft/speecht5_tts", use_safetensors=True)
        model = SpeechT5ForTextToSpeech.from_pretrained("/usr/src/app/models/microsoft/speecht5_tts", use_safetensors=True)
        vocoder = SpeechT5HifiGan.from_pretrained("/usr/src/app/models/microsoft/speecht5_hifigan", use_safetensors=True)


    else:
        print("No speecht5_tts saves, downloading and saving models")

        processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        processor.save_pretrained("/usr/src/app/models/microsoft/speecht5_tts", safe_serialization=True)
        model.save_pretrained("/usr/src/app/models/microsoft/speecht5_tts", safe_serialization=True)
        vocoder.save_pretrained("/usr/src/app/models/microsoft/speecht5_hifigan", safe_serialization=True)

        print("speecht5_tts saved")

     # Load a speaker embedding from the pre-trained dataset
    dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embedding = dataset[0]["xvector"]  # Use the first speaker embedding
    speaker_embedding = torch.tensor(speaker_embedding).unsqueeze(0)  # Convert to tensor and add batch dimension

    # Prepare input text
    inputs = processor(text=text, return_tensors="pt")

    # Generate speech
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings=speaker_embedding, vocoder=vocoder)

    # Convert the audio tensor to NumPy and get the sampling rate
    audio_waveform = speech.numpy()
    sampling_rate = 16000  # SpeechT5 outputs audio at 16kHz

    return {"samplerate": sampling_rate, "audio": audio_waveform.tolist()}

"""
@app.post("/save_model")
async def models(request: TextRequest):
"""