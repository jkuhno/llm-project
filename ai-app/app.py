import gradio as gr
import requests
import json
import numpy as np

### Lightweight UI for the assistant
### Requests predictions from ai-models API server
### Copyright Jani kuhno

def generate_response(audio, use_rag):
    # For future, currently does nothing
    print(f"RAG usage toggled: {use_rag}")
    with open(audio, 'rb') as f:
        response = requests.post("http://ai-models:8000/generate", files={"file": f})
    
    if response.status_code == 200:
        data = response.json()
        return data["response"] #, None, None
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

    return audio_tuple # answer,


#### UI #####
#############


theme = gr.themes.Soft()


with gr.Blocks(theme="YTheme/Minecraft") as app:

    gr.HTML(
        f"""
        <h1 style='text-align: center;'> LIZA  👩‍🔧</h1>
        <h3 style='text-align: center;'> The AI assistant from Temu </h3>
        """
    )
    with gr.Column():
        audio_in = gr.Audio(label="Speak your question", sources="microphone", type="filepath")
        # For future use
        use_rag = gr.Checkbox(label="Use retrieval memory")
        rec_btn = gr.Button(value="Send to models")
        answer = gr.Textbox(label="Answer as text")
        audio_out = gr.Audio(label="Spoken Answer", autoplay=True)

    rec_btn.click(fn=generate_response, inputs=[audio_in, use_rag], outputs=answer).then(
        fn=read_response, inputs=answer, outputs=audio_out)


app.launch(share=False)