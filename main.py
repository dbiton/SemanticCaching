import json
from random import sample
from typing import Dict, List
import faiss
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from cache_policy import *

def embed_strings(strings: List[str], model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(strings, convert_to_numpy=True)
    return embeddings.astype(np.float64)

def load_data_long(n: None):
    with open("prompts.json", "r") as f:
        data = json.load(f)
        origin = [v["text_a"] for v in data if v["label"] == 1]
        similar = [v["text_b"] for v in data if v["label"] == 1]
        if n:
            indices = sample(list(range(len(origin))), n)
            origin = [origin[i] for i in indices]
            similar = [similar[i] for i in indices]
        return origin, similar

def load_data_short():
    with open("mock_data.json", "r") as f:
        return json.load(f)

def load_quora():
    df = pd.read_csv('questions.csv')
    x = 3
    
def plot(results):
    """
    Plots the results of dimensionality reduction experiments.

    Args:
        results (dict): A dictionary where keys are reduction technique names and values are
                        lists of tuples (dim_reduce, score).
    """
    
    for prop_name, prop_results in results.items():
        plt.figure()

        i = 0
        markers = ["o", "s", "^", "v", "D"]
        for cache_name, prop in prop_results.items():
            cache_size = list(prop.keys())
            prop_values = list(prop.values())  # Unzip dimensions and scores
            plt.plot(cache_size, prop_values, label=cache_name, marker=markers[i])
            i += 1
        plt.xlabel("Cache Size")
        plt.ylabel(prop_name)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{prop_name}.png")

def main():
    load_quora()
    
    strings_origin, strings_similar = load_data_long(2000)
    embeds = embed_strings(strings_origin + strings_similar)
    embeds_origin = embeds[:len(strings_origin)]
    embeds_similar = embeds[len(strings_origin):]
    
    l2_distances = np.linalg.norm(embeds_origin - embeds_similar, axis=1)
    same_embed_distance = np.max(l2_distances)
    
    dim = embeds_origin[0].shape[0]
    
    policies = {
        "RR": RR,
        "LRU": LRU,
        "LFU": LFU,
        "FIFO": FIFO
    }
    
    results = {}
    
    for policy_name, policy_constructor in policies.items():
        for cache_size in range(32, 640, 32):
            index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
            policy: CachePolicy = policy_constructor(cache_size)
            cache_hits = 0
            for i_embed, embed in enumerate(embeds):
                embed_as_array = embed.reshape(1, embed.shape[0])
                distances, neighbors = index.search(embed_as_array, 1)
                if neighbors[0][0] != -1 and distances[0][0] <= same_embed_distance:
                    cache_hits += 1
                    embed_remove = policy.log_access(tuple(embed.flatten()))
                else:
                    index.add_with_ids(embed_as_array, np.array([i_embed]))
                    embed_remove = policy.log_access(tuple(embed.flatten()))
                if embed_remove is not None:
                    removed_id = np.where((embeds == np.array(embed_remove)).all(axis=1))[0][0]
                    index.remove_ids(np.array([removed_id]))
            iter_results = {
                "Cache Hit Percentage": cache_hits / len(embeds)
            }
            
            for prop_name, prop in iter_results.items():
                if prop_name not in results:
                    results[prop_name] = {}
                if policy_name not in results[prop_name]:
                    results[prop_name][policy_name] = {}
                results[prop_name][policy_name][cache_size] = prop
            print(policy_name, iter_results)
    plot(results)

if __name__=="__main__":
    main()