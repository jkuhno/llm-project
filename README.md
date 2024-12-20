# A personal assistant of sorts
A final product should be a voice assistant server running smooth on local machine, with access to bespoke tools.

The platform for running the solution is Docker.

#### Meta Llama

To use the assistant, you need access to Meta Llama gated models in Hugging Face hub.

Once granted, generate an [access token](https://huggingface.co/docs/hub/security-tokens) and store the token
in a text file called *hf_token.txt* inside the top level project folder.

#### LangSmith

LangSmith can be used for tracing by saving the API key as a text file *langsmith.txt* inside the top level project folder.

### This project is used for learning different AI types

- Speech recognition
- Text to speech
- Chat
- Inference on GPU

### Todo

- **Kinda done** Increase inference server performance
- **Kinda done, needs refinement** Fix the robotic voice
- Streaming
- Add agentic capability
- Create tools
- Switch tts to StyleTTS2
- Switch from Hugging Face to Ollama or vLLM (faster, according to the internet)


**Built with Llama**
