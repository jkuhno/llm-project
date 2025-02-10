from langchain_core.output_parsers import JsonOutputParser # type: ignore
from langchain_core.messages import AIMessageChunk # type: ignore
from langgraph.graph import START, END, MessagesState, StateGraph # type: ignore
from langchain_core.tools import tool # type: ignore
from langgraph.prebuilt import ToolNode # type: ignore
from langchain_core.runnables.config import RunnableConfig, ensure_config, get_callback_manager_for_config # type: ignore
from langchain_core.language_models.chat_models import ChatGenerationChunk # type: ignore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings # type: ignore

import requests # type: ignore
import os # type: ignore

from api_server.utils.db_client import ConnectPostgres
from api_server.utils import prompts
from api_server.models.vllm_chat import get_vllm_chat_model

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DIMS = 768
GMAPS_API_KEY = os.environ['GMAPS_API_KEY']

# Setup
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
chat_model = get_vllm_chat_model()

sync_connection = ConnectPostgres(embeddings, DIMS)

with sync_connection.get_store() as store:
    store.setup()

def get_graph(config: RunnableConfig) -> StateGraph: 
        

    ###################
    ###### GRAPH ######
    ###################

    # support function
    def get_reviews(id: str) -> str:
        # Before querying, check if store already has these id's
        # This could/should be a common database and not in user namespace
        # In order to make as much cached content avilable as possible
        with sync_connection.get_store() as store:
            stored_reviews = store.get(namespace=("restaurants", "reviews"), key=id)
        if stored_reviews:
            return stored_reviews.value["reviews"]
            
        url = f"https://places.googleapis.com/v1/places/{id}"
     
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": GMAPS_API_KEY,
            "X-Goog-FieldMask": "reviews.text.text",
        }
     
        response = requests.get(url, headers=headers)
        reviews = response.json()["reviews"][:2]
        
        formatted_list = []
        i = 1
        for rev in reviews:
            formatted_list.append(f'Review {i}: {rev["text"]["text"]}')
            i += 1
        print("Credits used: 2x Place Details (Preferred) SKU. Cost: 0.05 USD")
        with sync_connection.get_store() as store:
            store.put(("restaurants", "reviews"), id, {"reviews": formatted_list, "id": id})
        
        return formatted_list
    
    @tool
    def get_restaurants(query: str) -> str:
        """ Get a list of restaurant suggestions matching the query.
        The input is the way to match results with user request.
        Make sure the input is as descriptive as possible while keeping
        the input concise, 5 to 10 words."""
        # Start by querying plain ID's, because they cost 0.00 EUR
        url = "https://places.googleapis.com/v1/places:searchText"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": GMAPS_API_KEY,
            "X-Goog-FieldMask": "places.id,"
        }
        # Needs a better way to handle location bias
        data = {
            "textQuery": f"{query}, restaurant in helsinki",
            "pageSize": "7",
            "openNow": "false",
            "locationBias": {
              "circle": {
                "center": {"latitude": 60.18504951454597, "longitude": 24.952783788542657},
                "radius": 1000.0
              }
            }
        }
        response = requests.post(url, headers=headers, json=data)
        # If ID is already stored in the PGStore, get items by ID to save gmap calls
        # If not, query Place Details for cheaper calls
        answers = """"""
        for id in response.json()["places"]:
            with sync_connection.get_store() as store:
                saved_item = store.get(namespace=("restaurants", "info"), key=id['id'])
            if saved_item:
                answers += f'Name: {saved_item.value["name"]}' + '\n'
                answers += f'Address: {saved_item.value["address"]}' + '\n'
                answers += f'Reviews: {get_reviews(id["id"])}' + '\n\n'
            else:
                url = f"https://places.googleapis.com/v1/places/{id['id']}"
        
                headers = {
                    "Content-Type": "application/json",
                    "X-Goog-Api-Key": GMAPS_API_KEY,
                    "X-Goog-FieldMask": "displayName,formattedAddress",
                }
             
                response = requests.get(url, headers=headers)
                print("Credits used: 2x Place Details (Location Only) SKU. Cost: 0.01 USD")
                data = response.json()
                with sync_connection.get_store() as store:
                    store.put(("restaurants", "info"), 
                               id['id'], 
                               {"name": data["displayName"]["text"], 
                                "address": data["formattedAddress"] ,
                                "id": id['id']})
                
                answers += f'Name: {data["displayName"]["text"]}' + '\n'
                answers += f'Address: {data["formattedAddress"]}' + '\n'
                answers += f'Reviews: {get_reviews(id["id"])}' + '\n\n'
        
        return answers
    
    @tool
    def upsert_preference(preference: str, emotion: str, config: RunnableConfig) -> str:
        """Save a user preference into memory. Can be positive or negative thoughts about things."""
        with sync_connection.get_store() as store:
            store.put(
                    (config["configurable"]["user_id"], "preferences"),
                     key=config["configurable"]["mem_key"],
                     value={"preference": preference, "emotion": emotion},
                    )
        return "Saved a preference!"
    

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

    
    workflow = StateGraph(State)
    
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