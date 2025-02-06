# A personal restaurant recommender
Previously experimented on short- and long-term memory in Langgraph and inference on HuggingFace.

Now focusing on making an actually helpful system, that can recommend places to it given user cravings and make a decision if user is indecisive.

Testing is done on laptop RTX 5000 Ada **16GB**, trying to keep the VRAM footprint small though.
The platform for running the solution is Docker.

#### Meta Llama

To use the assistant, you need access to Meta Llama gated models in Hugging Face hub.

Once granted, generate an [access token](https://huggingface.co/docs/hub/security-tokens) and store the token in the .env file, see below.

#### Configure .env

Env variables for the project setup are configured by creating a file called '.env' in the project folder, with the following contents:
```
HF_TOKEN=<your-huggingface-hub-token>
GMAPS_API_KEY=<your-google-maps-api-key>
LANGCHAIN_API_KEY=<your-langsmith-token> # optional, remove from yaml if not used
POSTGRES_PASSWORD=<create-password-of-choice>
POSTGRES_USER=<create-username-of-choice>
POSTGRES_DB='index.db' 
```

Make sure to keep the env file in gitignore. In there by default.


#### Running

`docker compose -f docker-compose-dev.yaml up --build` 

or

`docker compose -f docker-compose-prod.yaml up --build`

Composing with *dev* has development features enabled, such as hot reload for frontend and API server,

while *prod* is slightly optimized to run as release. *prod* runs the frontend with Nginx, 

API server with uvicorn in release mode and ollama with GIN_MODE as release.

*New option (not finalized yet)*

`docker compose -f docker-compose-langgraph.yaml up --build` 

Launches the project as a [LangGraph development server](https://langchain-ai.github.io/langgraph/concepts/langgraph_cli/#dev)

It can be connected to with LangGraph Studio by 

`https://smith.langchain.com/studio/?baseUrl=http://localhost:2024`

Because currently the graph uses a custom function for chat calls directly to ollama server, the main chat 
node does not display correctly in LangGraph Studio,

which is annoying since that is the main node to debug. Will be fixed.


**Built with Llama**
