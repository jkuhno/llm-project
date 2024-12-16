import gradio as gr
import requests
import json
import numpy as np

### Lightweight UI for the assistant
### Requests predictions from ai-models API server
### Copyright Jani kuhno

def generate_response(audio):
    with open(audio, 'rb') as f:
        response = requests.post("http://ai-models:8000/generate", files={"file": f})
    
    if response.status_code == 200:
        data = response.json()
        return data["response"], None, None
    else:
        return f"{response.status_code}: There was an error generating the answer"


def read_response(answer):
    response = requests.post("http://ai-models:8000/speech", 
                              json={"text": answer},

                            )
    data = response.json()
    sampling_rate = data["samplerate"]
    audio = np.array(data['audio'], dtype=np.float32)
    audio_tuple = (sampling_rate, audio)

    return answer, audio_tuple


with gr.Blocks() as app:
    gr.HTML(
        f"""
        <h1 style='text-align: center;'> Jarvis  🤵</h1>
        <h3 style='text-align: center;'> The AI assistant from Temu </h3>
        """
    )
    with gr.Group():
        with gr.Row():
            audio_out = gr.Audio(label="Spoken Answer", autoplay=True)
            answer = gr.Textbox(label="Answer")
            state = gr.State()
        with gr.Row():
            audio_in = gr.Audio(label="Speak your question", sources="microphone", type="filepath")


    audio_in.stop_recording(generate_response, audio_in, [state, answer, audio_out])\
        .then(fn=read_response, inputs=state, outputs=[answer, audio_out])\


app.launch(share=False)