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
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan



### API server housing AI models
### Serving model outputs via FastAPI
### Copyright Jani Kuhno

class TextRequest(BaseModel):
    text: str


class AudioResponse(BaseModel):
    samplerate: int
    audio: List[float]  # Audio waveform as a list of floats


app = FastAPI()


def generate_answer(transcription):

    pipe = pipeline("text-generation", "stabilityai/stablelm-zephyr-3b", device="cuda:0", torch_dtype=torch.bfloat16)
    messages = [
        {
        "role": "system",
        "content": "You are a friendly assistant who always responds in ten words or less",
        },
        {"role": "user", "content": transcription},
    ]
    return pipe(messages, max_new_tokens=64)[0]['generated_text'][-1]["content"]

"""
    model_id = "gpt2"
    generation_pipeline = pipeline("question-answering", model=model_id, device="cuda:0") # model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
    answer = generation_pipeline(question=transcription, context="I've been quite good lately. Been working out and getting a lot of sun.")
    print(answer)
    return answer["answer"]
"""

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

    # Load the processor and model
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

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