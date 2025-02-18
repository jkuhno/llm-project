from langchain_core.tools import tool # type: ignore
import os
import requests # type: ignore
from langchain_core.runnables.config import RunnableConfig # type: ignore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings # type: ignore

from api_server.utils.db_client import ConnectPostgres

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DIMS = 768
GMAPS_API_KEY = os.environ['GMAPS_API_KEY']

# Setup
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

sync_connection = ConnectPostgres(embeddings, DIMS)

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