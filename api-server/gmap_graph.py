from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessageChunk
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.runnables.config import RunnableConfig, ensure_config, get_callback_manager_for_config
from langchain_core.language_models.chat_models import ChatGenerationChunk

from db_client import ConnectPostgres

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
        """
        # support function
        def get_reviews(id: str) -> str:
            gmaps = googlemaps.Client(key=GMAPS_API_KEY)
    
            place = googlemaps.places.place(
                client=gmaps,
                place_id=id,
                fields=["reviews"],
                reviews_no_translations=False,
                reviews_sort="most_relevant",
            )
            keys = ["text"]
            reviews = [{key: result[key] for key in keys if key in result} for result in place["result"]["reviews"]]
            return reviews[0]["text"]

       
        @tool
        def get_restaurants(query: str) -> list[dict]:
            
            gmaps = googlemaps.Client(key=GMAPS_API_KEY)
            
            places = googlemaps.places.places(
                client=gmaps,
                query=query,
                location=(60.18504951454597, 24.952783788542657), # LatLong or human-readable address
                radius=1000, # in meters
                open_now=True,
            )
            
            keys_to_keep = ["name", "place_id", "formatted_address", "rating"]
            formatted_results = [{key: result[key] for key in keys_to_keep if key in result} for result in places["results"]]
            sorted_places = sorted(formatted_results[:7], key=lambda x: x['rating'], reverse=True)
            
            for place in sorted_places:
                place["reviews"] = get_reviews(place["place_id"])
            
            return sorted_places
        """
        
        @tool
        def get_restaurant(query: str, config: RunnableConfig) -> list[dict]:
            """ Get a list of restaurant suggestions matching the query """
            with self.sync_connection.get_store() as store:
                items = store.search(
                    (config["configurable"]["user_id"], "preferences")
                )
            return [{"name": "Nabi Korean BBQ", "place_id": "ChIJgYslZf4LkkYRA0gWJ4zFWIE", "formatted_address": "Eerikinkatu 14, 00100 Helsinki, Finland", "rating": 4.9, 
              "reviews": ("Very authentic Korean BBQ experience! We went all in with a full menu of assorted meats and seafood. The marinated meats naturally "
              "were most flavorful! Alex was our host and with his friendly attitude and attention to detail he really made this an unforgettable evening! "
              "He explained every dish and thought us how to grill the food. If you are interested in an interactive and delicious dinner experience Nabi is "
              "the perfect choice!\\nWe will be back soon for more 😍")
              }, 
             {"name": "Oppa Korean BBQ Kluuvi", "place_id": "ChIJW4LQBNkLkkYRzBtBndafJhc", "formatted_address": "Yliopistonkatu 6, 00100 Helsinki, Finland", "rating": 4.7, 
              "reviews": ("Review of Oppa Korean BBQ in Helsinki, Finland\\n\\nRecently, we visited Oppa Korean BBQ in Helsinki, Finland, and the experience was absolutely unforgettable. "
                          "This restaurant is a true slice of South Korea, where you can not only enjoy the most delicious dishes but also immerse yourself in an authentic atmosphere.\\n\\nThe "
                          "highlight of the restaurant is the built-in grills at each table. The opportunity to grill your own meat is not just about cooking—it’s an interactive experience that "
                          "transforms your meal into a culinary event. The quality of the products is exceptional—every piece of meat is fresh and of the highest grade. Even the most luxurious and "
                          "exquisite option, Hanwoo beef (renowned for its rich flavor and marbled texture), is top-notch here.\\n\\nBeyond the meat, the side dishes and seasonings are truly impressive. "
                          "The traditional kimchi is spicy, crispy, and absolutely perfect. The spices are expertly selected, enhancing the flavor of every dish and making each bite more vibrant and "
                          "memorable.\\n\\nThe service deserves special praise. The staff is incredibly responsive, attentive, and genuinely welcoming. They are always ready to assist, explain the menu, "
                          "and even guide you on how to grill the meat to bring out its best flavors. Such thoughtful service adds warmth and comfort to the dining experience.\\n\\nThe restaurant’s ambiance "
                          "is meticulously designed: a stylish interior with touches of Korean culture, cozy lighting, and a harmonious atmosphere make it an ideal spot for meeting friends or enjoying a "
                          "family dinner.\\n\\nIf you’re looking to savor the authentic flavors of South Korea and indulge in the freshest ingredients, Oppa Korean BBQ in Helsinki is the place to be. This "
                          "restaurant will leave you with lasting impressions and might just become your new favorite. Highly recommended to anyone who appreciates top-quality food, excellent service, "
                          "and a unique dining atmosphere.")
             }, 
             {"name": "Chingu Korean BBQ Helsinki Kalasatama", "place_id": "ChIJBW6sVysJkkYRJKXz7uO-Yc8", "formatted_address": "Tukkutorinkuja 2 C, 00580 Helsinki, Finland", "rating": 4.7, 
              "reviews": ("The meat cuts are good. They have a wide selection of food items and hot pot service. The deserts section is mediocre. And you have unlimited tea and coffee as well. What really "
                          "stood out of this restaurant is The service is top notch. We have travelled more than 33 countries been to many restaurants. Alicia is an asset to the restaurant for her incredible "
                          "passion and love for job. We felt really taken care of.")
             }, 
             {"name": "Shabu House Korean BBQ Restaurant", "place_id": "ChIJ_4kEwCQJkkYR9K7SuQClL6s", "formatted_address": "Hämeentie 31, 00500 Helsinki, Finland", "rating": 4.6, 
              "reviews": ("My friend introduced me to this place and I fell in love with it. The atmosphere was cozy and the food was incredible. High quality meat that you could cook by yourself to how ever "
                          "you wanted gave this 5 stars in my opinion. The waiter was professional and he was really kind. I would definitely come here again and I recommend this place!")
             }, 
             {"name": "Oppa Korean BBQ Redi", "place_id": "ChIJ693sgdgJkkYRDZ3oLtSCaeI", "formatted_address": "Hermannin rantatie 5, 00580 Helsinki, Finland", "rating": 4.5, 
              "reviews": ("It’s a really nice place. The food presentation is very nice. It is very Korean style barbecue restaurant . And very good service! Three of us ordered two sets. "
                          "The waiter gave us extra desserts and sesame and seaweed rice as gifts. They are so friendly here.")
             }, 
             {"name": "Kimchi Korean BBQ Buffet ,Kamppi", "place_id": "ChIJL7fnDQILkkYRp-Wk4FPdvw0", "formatted_address": "Fredrikinkatu 49, 00100 Helsinki, Finland", "rating": 4.4, 
              "reviews": ("Let’s start with the service as it was outstanding, friendly, polite, talkative if liked. We knew HotPot and Thai BBQ, were asked if we want a quick introduction. "
                          "Additionally, you find a short video on a TV showing you some guides how to do Korean BBQ.\\n\\nThe food was delicious. Having an outstanding BBQ we didn’t expect the "
                          "Sushi being so delicious as well. I simply don’t know which meat, fish or vegetables to mention first as they all were fresh and perfect in taste.\\n\\nWe visited the "
                          "restaurant when there was an additional 20% discount on BBQ and we paid less than 30€ including drinks per person. Just awesome.")
             }, 
             {"name": "Happy Food Garden", "place_id": "ChIJ8z2kHssLkkYR1vj8TQ-KY50", "formatted_address": "Kalevankatu 23, 00100 Helsinki, Finland", "rating": 4, 
              "reviews": "One of my favorites place in Helsinki!!! The service is kind and very efficient nd good is the top, especially hot pot, barbecue and the lemon sauce chicken fried 😱😍😍😍😍🥰🥰🥰🥰🤩🤩🤩🤩🤩🤩🤩👏👏👏👏🙏"},
             {"user_preferences": items}]

        
        @tool
        def upsert_preference(preference: str, config: RunnableConfig) -> str:
            """Save a user preference into memory. Can be positive or negative thoughts about things."""
            with self.sync_connection.get_store() as store:
                store.put(
                        (config["configurable"]["user_id"], "preferences"),
                         key=config["configurable"]["mem_key"],
                         value={"text": preference},
                        )
            return f"Saved a preference: {preference}"

        
        tools = [get_restaurant, upsert_preference]
        tool_node = ToolNode(tools)
        llm_with_tools = self.chat_model.bind_tools(tools)


        class State(MessagesState):
            router: str
        
        
        def router(state: State, config: RunnableConfig):
            messages = state["messages"][-1]
            invoker = {"input": messages,
                       "tools": tools,
                  }
            prompt_template = PromptTemplate(
                template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|>
                         You are a binary router. You get a user input, a list of tools and decide if the input needs tool calls.
                         Tool calls are the right option, IF the user input context matches any of the function descriptions in tools provided 
                         AND you cannot answer to the input without help. In case you decide tool calls are needed, answer value is "yes".
                         Functions for tool calls are here: [{tools}].
                         If no tool calls need to be made, as in the input can be answered without the functions, answer value is "no".
        
                         ALWAYS answer in JSON-like format with the key being "router" and the value being "yes" or "no": {{"router": "value"}}
                         DO NOT include anything else in your answer but the JSON-like key-value pair
        
                         *** EXAMPLES ***
                         User input: 'Hi!'
                         Your answer: {{"router": "no"}}
        
                         User input: 'I want to eat something fresh'
                         Your answer: {{"router": "yes"}}
                         
                         <|eot_id|>
                         <|start_header_id|>user<|end_header_id|>
                         Here is the user input: {input}
                         <|eot_id|>
                         <|start_header_id|>assistant<|end_header_id|>""",
                input_variables=["input", "tools"],
            )
            
            rag_chain = prompt_template | llm_with_tools | JsonOutputParser()
            response = rag_chain.invoke(invoker)
            return response

        def router_conditional(state: State, config: RunnableConfig):
            #print(state)
            if state["router"] == "yes":
                return "agent"
            elif state["router"] == "no":
                return "chat"
            else:
                #state["messages"] + [f"Error occurred in routing. The routing value {state['router']} is not accepted"]
                return "chat"
        
        
        def agent(state: State, config: RunnableConfig):
            messages = state["messages"]
            invoker = {"input": messages,
                       "tools": tools,
                  }
            prompt_template = PromptTemplate(
                template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|>
                         You are a diligent agent making function calls. Your task is to make a call to a function that best matches the context of 
                         the user input.
                         Infer possible parameters from the user question and use them in a way that meets the function requirements.
        
                         Here are the functions: {tools}
                         <|eot_id|>
                         <|start_header_id|>user<|end_header_id|>
                         Here is the input: {input}
                         <|eot_id|>
                         <|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["input", "tools"],
            )
            
            prompt = prompt_template.invoke(invoker)
            response = llm_with_tools.invoke(prompt)
            return {"messages": [response]}
        
        
        def chat(state: State, config: RunnableConfig):
            messages = state["messages"]
            config = ensure_config(config | {"tags": ["chat_llm"]})
            callback_manager = get_callback_manager_for_config(config)
        
            llm_run_manager = callback_manager.on_chat_model_start({}, [messages])[0]
            
            client = self.direct_ollama_model
        
            user_input = messages[0]
            tool_msg = messages[-1]
            # if first and the last message are the same, this means it came straight from the router
            if user_input == tool_msg:
                invoker = {"input": user_input.content, "tool_msg": ""}
            else:
                invoker = {"input": user_input.content, "tool_msg": f"To assist you in responding, \
                here is an output from an internal function call that is related to the user input: {tool_msg.content} \
                Use this output to respond to the user input"}
            
            prompt = [
                {
                  "role": "system",
                  "content": f"You are a helpful assistant. Respond to the input politely. {invoker['tool_msg']}"
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
                response_content += tokens['message']['content']
                chunk = ChatGenerationChunk(
                        message=AIMessageChunk(
                            content=tokens['message']['content'],
                        )
                )
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
        workflow.add_node("agent", agent)
        workflow.add_node("chat", chat)
        workflow.add_node("tools", tool_node)
        
        workflow.add_edge(START, "router_model")
        workflow.add_conditional_edges("router_model", router_conditional, ["agent", "chat"])
        workflow.add_edge("agent", "tools")
        workflow.add_edge("tools", "chat")
        workflow.add_edge("chat", END)

        #checkpointer = MemorySaver()
        app = workflow.compile()
        return app