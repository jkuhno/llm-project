# A personal assistant of sorts
A final product should be a voice assistant server running smooth on local machine, with access to bespoke tools.

Currently can listen to human speech, chat with short- and long-term memory and produce the output as speech.

Short-term memory is in-memory, long-term memory is presisted in postgres.
The long-term memory is saved in a parallel node with the chat model, but has very crude handling for duplicates and no debouncing.

Testing is done on Quadro RTX 4000 **8GB**, should fit in 6 or less with a little more optimization.

The platform for running the solution is Docker.

#### Meta Llama

To use the assistant, you need access to Meta Llama gated models in Hugging Face hub.

Once granted, generate an [access token](https://huggingface.co/docs/hub/security-tokens) and store the token
in a text file called *hf_token.txt* inside the top level project folder.

#### Postgres

Save postgres username and password as respective text files *db_user.txt* and *db_pass.txt* inside the top level project folder.

#### LangSmith

LangSmith can be used for tracing by saving the API key as a text file *langsmith.txt* inside the top level project folder.


### Todo

- **Kinda done** Increase inference server performance
- **Kinda done, needs refinement** Fix the robotic voice
- Streaming
- Add agentic capability (*1B and 3B Llama models require custom templates for tools in LangChain*)
- Switch tts to StyleTTS2
- Switch from Hugging Face to Ollama or vLLM (faster, according to the internet)


**Built with Llama**
