from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langgraph.graph import START, END, MessagesState, StateGraph

from db_client import ConnectPostgres, PoolConfig

class SyncGraph:
    def __init__(self, embeddings, dims, trimmer, chat_model):
        sync_connection = ConnectPostgres(embeddings, dims)
        self.sync_connection = sync_connection
        self.trimmer = trimmer
        self.chat_model = chat_model
        

    def get_graph(self):
        with self.sync_connection.get_store() as store:
            store.setup()

        def chat(state, *, store: BaseStore, config):
            # Search based on user's last message
            user_id = config["configurable"]["user_id"]

            with self.sync_connection.get_store() as store:
                items = store.search(
                    (user_id, "memories"), query=state["messages"][-1].content, limit=4, offset=1
                )

            memories = "\n".join(item.value["text"] for item in items)
            memories = f"## Memories of user: {memories}" if memories else ""

            prompt_template = ChatPromptTemplate.from_messages(
                [
                (
                   "system",
                    ("You are a helpful assistant, who always greets the user with the word 'sir'. "
                     "Answer concisely to any request or question provided by the user. Use fifteen words or less. "
                     "If answering in numbers, use written form. For example: answer 'number ten' and not 'number 10'"
                      "You are provided with memories related to the user request. Use the memories for context if needed"
                      "{memory}"
                      "Do not directly cite these memories, unless told so."
                    ),
                ),
                    MessagesPlaceholder(variable_name="messages"),
                ],
            )
            prompt_template = prompt_template.partial(memory=memories)
        
            trimmed_messages = self.trimmer.invoke(state["messages"])
            prompt = prompt_template.invoke(trimmed_messages)
            print(prompt)
            response = self.chat_model.invoke(prompt)
        
            return {"messages": [response]}
    
    
        # node to save user input into long-term memory
        # crude duplicate check, needs imporvement
        def save_memories(
            state,
            *,
            store: BaseStore,
            config
        ):
            user_id = config["configurable"]["user_id"]
            memory = state["messages"][-1].content
            mem_key = config["configurable"]["mem_key"]

            with self.sync_connection.get_store() as store:
                items = store.search(
                    (user_id, "memories"), query=state["messages"][-1].content, limit=10
                )

            memories = "\n".join(item.value["text"] for item in items)
            if memory not in memories:
                with self.sync_connection.get_store() as store:
                    store.put(
                        (user_id, "memories"),
                         key=mem_key,
                         value={"text": memory},
                        )
            else:
                print("duplicate")
    
        builder = StateGraph(MessagesState)
        builder.add_node(chat)
        builder.add_node(save_memories)
        builder.add_edge(START, "chat")
        builder.add_edge(START, "save_memories")
        builder.add_edge("chat", END)

        checkpointer = MemorySaver()
        graph = builder.compile(checkpointer=checkpointer, store=store)
        return graph