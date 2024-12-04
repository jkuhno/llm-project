import gradio as gr
import requests
import json
import numpy as np

### Lightweight UI for the assistant
### Requests predictions from ai-models API server
### Copyright Jani kuhno

def generate_response(audio):
    #print(audio)
    with open(audio, 'rb') as f:
        response = requests.post("http://ai-models:8000/upload", files={"file": f})
        
    return response.text, None, None


def read_response(answer):
    response = requests.post("http://ai-models:8000/generate", 
                              json={"text": answer},

                            )
    data = response.json()
    sampling_rate = data["samplerate"]
    audio = np.array(data['audio'], dtype=np.float32)
    audio_tuple = (sampling_rate, audio)

    return answer, audio_tuple


def save_model(model_name):
    response = requests.post("http://ai-models:8000/save_model", 
                              json={"text": model_name},

                            )
    message = response.json()
    return message


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
    with gr.Row():
        model = gr.Textbox(label="Check if HF model is saved, save if not")
        model_btn = gr.Button("Check / Save")


    audio_in.stop_recording(generate_response, audio_in, [state, answer, audio_out])\
        .then(fn=read_response, inputs=state, outputs=[answer, audio_out])\

    model_btn.click(fn=save_model, inputs=model, outputs=model)
app.launch(share=False)