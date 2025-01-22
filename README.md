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
`
HF_TOKEN=<your-huggingface-hub-token>
GMAPS_API_KEY=<your-google-maps-api-key>
LANGCHAIN_API_KEY=<your-langsmith-token>
POSTGRES_PASSWORD=<create-password-of-choice>
POSTGRES_USER=<create-username-of-choice>
POSTGRES_DB='index.db' # optional, remove from yaml if not used
`

Make sure to keep the env file in gitignore. In there by default.



**Built with Llama**
