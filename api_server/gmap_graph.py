from langgraph.graph import START, END, MessagesState, StateGraph # type: ignore
from langchain_core.tools import tool # type: ignore
from langgraph.prebuilt import ToolNode # type: ignore
from langchain_core.runnables.config import RunnableConfig # type: ignore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings # type: ignore

import requests # type: ignore
import os # type: ignore

from api_server.utils.db_client import ConnectPostgres
from api_server.utils import prompts
from api_server.models.vllm_chat import get_vllm_chat_model
from api_server.configuration import Configuration
from api_server.utils.tools import get_restaurants, upsert_preference


EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DIMS = 768
GMAPS_API_KEY = os.environ['GMAPS_API_KEY']

# Setup
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
chat_model = get_vllm_chat_model()

sync_connection = ConnectPostgres(embeddings, DIMS)

with sync_connection.get_store() as store:
    store.setup()


# A function to define the graph, for seamless use in LangSmith Studio
def get_graph(config: RunnableConfig) -> StateGraph: 
    tools = [get_restaurants, upsert_preference]
    tool_node = ToolNode(tools)
    llm_with_tools = chat_model.bind_tools(tools)


    class State(MessagesState):
        router: str
    
    
    def router(state: State, config: RunnableConfig):
        messages = state["messages"][-1]
        invoker = {"input": messages,
                   "tools": tools,
              }
        # Input variables populate associated keys in the prompt
        # They are usually, like here, the keys of the invoker dict
        prompt_template = prompts.router_prompt(["input", "tools"])

        json_schema = {
            "title": "Router",
            "description": "Routing to decide which tool to call, if any is necessary.",
            "type": "object",
            "properties": {
                "router": {
                    "type": "string",
                    "enum": ["maps", "save", "no"],
                }
            },
        }
        
        chain = prompt_template | llm_with_tools.with_structured_output(json_schema)
        response = chain.invoke(invoker)
        return {"router": response["router"]}
    

    def router_conditional(state: State, config: RunnableConfig):
        if state["router"] == "maps":
            return "maps_agent"
        elif state["router"] == "save":
            return "save_agent"
        elif state["router"] == "no":
            return "chat"
        else:
            state["messages"] + [f"Error occurred in routing. The routing value {state['router']} is not accepted"]
            return "chat"
            
    
    def maps_agent(state: State, config: RunnableConfig):
        messages = state["messages"]
        invoker = {"input": messages,
                   "tools": tools[0],
              }
        prompt_template = prompts.maps_agent_prompt(["input", "tools"])
        
        prompt = prompt_template.invoke(invoker)
        response = llm_with_tools.invoke(prompt)
        return {"messages": [response]}
    
    def save_agent(state: State, config: RunnableConfig):
        messages = state["messages"]
        invoker = {"input": messages,
                   "tools": tools[1],
              }
        prompt_template = prompts.save_agent_prompt(["input", "tools"])
        
        prompt = prompt_template.invoke(invoker)
        response = llm_with_tools.invoke(prompt)
        return {"messages": [response]}
    
    
    def chat(state: State, config: RunnableConfig):
        messages = state["messages"]
        with sync_connection.get_store() as store:
            memories = store.search((config["configurable"]["user_id"], "preferences"))
            
        preferences = []
        for memory in memories:
            if "preference" in memory.value.keys():
                preferences.append(memory.value["preference"])
                
        user_input = messages[0]
        tool_msg = messages[-1]
            
        # chat_prompt already invokes the PrompTemplate
        prompt = prompts.chat_prompt(user_input, tool_msg, preferences)
        response = chat_model.invoke(prompt)
        return {"messages": [response]}

    
    workflow = StateGraph(State, config_schema=Configuration)
    
    # Define the two nodes we will cycle between
    workflow.add_node("router_model", router)
    workflow.add_node("maps_agent", maps_agent)
    workflow.add_node("save_agent", save_agent)
    workflow.add_node("chat", chat)
    workflow.add_node("tools", tool_node)
    
    workflow.add_edge(START, "router_model")
    workflow.add_conditional_edges("router_model", router_conditional, ["maps_agent", "save_agent", "chat"])
    workflow.add_edge("maps_agent", "tools")
    workflow.add_edge("save_agent", "tools")
    workflow.add_edge("tools", "chat")
    workflow.add_edge("chat", END)
    #checkpointer = MemorySaver()
    app = workflow.compile()
    app.name = "gmap_graph"
    return app