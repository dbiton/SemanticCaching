import json
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from vector_store import VectorStore

def embed_strings(strings: List[str], model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(strings, convert_to_numpy=True)
    return embeddings

def load_data():
    with open("mock_data.json", "r") as f:
        return json.load(f)

def main():
    data = load_data()
    strings_origin = [v['origin'] for v in data]
    strings_similar = [v['similar'] for v in data]
    embeds_origin = embed_strings(strings_origin)
    embeds_similar = embed_strings(strings_similar)
    dim = embeds_origin[0].shape[0]
    store = VectorStore(dim)
    store.add(embeds_origin, strings_similar)
    strings_pred, embeds_pred, _ = store.search(embeds_similar, 1)
    for i, (embed_pred, embed_actual) in enumerate(zip(embeds_pred, embeds_origin)):
        if not np.array_equal(embed_pred, embed_actual):
            print(f"({i})Failed in finding most similar!")
            print("Actual:", strings_origin[i])
            print("Pred:", strings_pred[i])
   
if __name__=="__main__":
    main()