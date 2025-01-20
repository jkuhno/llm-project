# A personal restaurant recommender
Previously experimented on short- and long-term memory in Langgraph and inference on HuggingFace.

Now focusing on making an actually helpful system, that can recommend places to it given user cravings and make a decision if user is indecisive.

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





**Built with Llama**
