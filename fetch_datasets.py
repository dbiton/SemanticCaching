import pickle
from typing import List
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

def embed_strings(strings: List[str], model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(strings, convert_to_numpy=True)
    return embeddings

if __name__ == "__main__":
    print("generating bing...")
    with open("embeds_bing.pkl", "wb") as f:
        ds_bing = load_dataset("corbyrosset/researchy_questions")
        questions_bing = ds_bing['train']['question']
        embeds_bing = embed_strings(questions_bing)
        pickle.dump(embeds_bing, f)

    print("generating so...")
    with open("embeds_so.pkl", "wb") as f:
        ds_so = load_dataset("pacovaldez/stackoverflow-questions")
        questions_so = ds_so['train']['title']
        embeds_so = embed_strings(questions_so)
        pickle.dump(embeds_so, f)

    print("generating chat...")
    with open("embeds_chat.pkl", "wb") as f:
        ds_chat = load_dataset("allenai/WildChat-1M")
        questions_chat = [e['conversation'][0]['content'] for e in ds_chat['train'] if e['language'] == "English"]
        embeds_chat = embed_strings(questions_chat)
        pickle.dump(embeds_chat, f)
