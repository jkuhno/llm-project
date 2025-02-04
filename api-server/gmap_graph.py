from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessageChunk
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.runnables.config import RunnableConfig, ensure_config, get_callback_manager_for_config
from langchain_core.language_models.chat_models import ChatGenerationChunk

import requests

from db_client import ConnectPostgres
import prompts

class GmapGraph:
    def __init__(self, 
                 embeddings, 
                 dims, 
                 trimmer, 
                 chat_model, 
                 direct_ollama_model, 
                 ollama_pulled_model,
                 api_key):
        
        sync_connection = ConnectPostgres(embeddings, dims)
        self.sync_connection = sync_connection
        self.trimmer = trimmer
        self.chat_model = chat_model
        self.direct_ollama_model = direct_ollama_model
        self.ollama_pulled_model = ollama_pulled_model
        self.api_key = api_key


    def get_graph(self):
        with self.sync_connection.get_store() as store:
            store.setup()
            
        GMAPS_API_KEY = self.api_key

        # support function
        def get_reviews(id: str) -> str:
            # Before querying, check if store already has these id's
            # This could/should be a common database and not in user namespace
            # In order to make as much cached content avilable as possible
            with self.sync_connection.get_store() as store:
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
            with self.sync_connection.get_store() as store:
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
                with self.sync_connection.get_store() as store:
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
                    with self.sync_connection.get_store() as store:
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
            with self.sync_connection.get_store() as store:
                store.put(
                        (config["configurable"]["user_id"], "preferences"),
                         key=config["configurable"]["mem_key"],
                         value={"preference": preference, "emotion": emotion},
                        )
            return "Saved a preference!"

        
        tools = [get_restaurants, upsert_preference]
        tool_node = ToolNode(tools)
        llm_with_tools = self.chat_model.bind_tools(tools)


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
            
            rag_chain = prompt_template | llm_with_tools | JsonOutputParser()
            response = rag_chain.invoke(invoker)
            return response

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
            config = ensure_config(config | {"tags": ["chat_llm"]})
            callback_manager = get_callback_manager_for_config(config)
        
            llm_run_manager = callback_manager.on_chat_model_start({}, [messages])[0]
            
            client = self.direct_ollama_model
        
            with self.sync_connection.get_store() as store:
                memories = store.search(("user_1", "preferences"), query="pizza")
            preferences = []
            for memory in memories:
                if "preference" in memory.value.keys():
                    preferences.append(memory.value["preference"])
                    
            user_input = messages[0]
            tool_msg = messages[-1]
            # if first and the last message are the same, this means it came straight from the router
            if user_input == tool_msg:
                invoker = {"input": user_input.content, "tool_msg": "Respond to the user normally and helpfully"}
            elif tool_msg.name == "get_restaurants":
                invoker = {"input": user_input.content, "tool_msg": f"""As an assistant, your role is 
                to help the user decide where to eat. You are given seven options, each 
                following the format Name, Address and two reviews. Here are the options: {tool_msg.content} 
                Use these options to present the user with your top 3 choices, always paying attention to user 
                preferences: {preferences}. From the top 3, suggest one which you think is the best match to 
                the user inquiry. Give a short explanation why you suggest that one. You do not need to copy all 
                text from the options, most important are restaurant names and why you pick them."""}
            else:
                invoker = {"input": user_input.content, "tool_msg": f"To assist you in responding, \
                here is an output from an internal function call that is related to the user input: {tool_msg.content} \
                Use this output to respond to the user input"}
            
            prompt = [
                {
                  "role": "system",
                  "content": f"You are a helpful assistant. Respond to the input using the following guidance: \
          {invoker['tool_msg']}"
                },
                {
                  "role": "user",
                  "content": invoker['input']
                },
            
            ]
            
            response = client.chat(model=self.ollama_pulled_model, 
                                   messages=prompt,
                                   stream=True
            )
            
            response_content = ""
            for tokens in response:
                #print(chunk['message']['content'], end='', flush=True)
                response_content += tokens['message']['content']
                chunk = ChatGenerationChunk(
                        message=AIMessageChunk(
                            content=tokens['message']['content'],
                        )
                )
                #print(chunk)
                llm_run_manager.on_llm_new_token(tokens['message']['content'], chunk=chunk)
                
            response_message = {
                "role": "assistant",
                "content": response_content,
                #"tool_calls": tool_calls,
            }
            return {"messages": [response_message]}

        
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
        return app