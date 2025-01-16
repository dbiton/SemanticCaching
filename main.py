import json
from random import sample
from typing import Dict, List
import faiss
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from cache_policy import *

has_gpu = False

def embed_strings(strings: List[str], model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(strings, convert_to_numpy=True)
    return embeddings

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

def embed_titles(n = -1):
    with open(f"Titles.txt", encoding="utf-8") as f:
        titles_strings = f.readlines(n)
    titles_embeddings = [e.tolist() for e in embed_strings(titles_strings)]
    with open("TitlesEmbeddings.json", "w") as f:
        json.dump(list(zip(titles_strings, titles_embeddings)), f)
    
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
        markers = ["o", "s", "^", "v", "D", "<", ">", "X", "+"]
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

def generate_embeds(mean, std_dev, dim, len):    
    np.random.seed(0)
    return [np.random.normal(mean, std_dev, dim) for _ in range(len)]

def load_embeds():
    with open("TitlesEmbeddings.json", "r") as f:
        embeds = json.load(f)
        return np.stack([np.array(e[1]) for e in embeds], axis=0)

def main():
    '''
    strings_origin, strings_similar = load_data_long(EMBEDS_COUNT)
    embeds = embed_strings(strings_origin + strings_similar)
    embeds_origin = embeds[:len(strings_origin)]
    embeds_similar = embeds[len(strings_origin):]
    
    l2_distances = np.lina5lg.norm(embeds_origin - embeds_similar, axis=1)
    same_embed_distance = np.max(l2_distances)
    '''
    print("loading embeds...")
    num_samples = 4000
    strings_origin, strings_similar = load_data_long(num_samples)
    embeds = embed_strings(strings_origin + strings_similar)
    sampled_indices = np.random.choice(embeds.shape[0], size=num_samples, replace=False)
    embeds = embeds[sampled_indices]
    print("loaded!")
    dim = 384
    same_embed_distance = 0.5
    similar_embed_distance = 0.707
    alpha = same_embed_distance / dim
    
    policies = {
        "ProximityScore": ProximityScore,
        #"ProbMisses": ProbMisses,
        #"ProbMinCounter": ProbMinCounter,
        #"ProbMinDensity": ProbMinDensity,
        "MinCounter": MinCounter,
        #"MinDensity": MinDensity,
        #"MaxDensity": MaxDensity,
        #"RR": RR,
        #"LRU": LRU,
        "LFU": LFU,
        # "FIFO": FIFO,
    }
    
    results = {}
    
    for policy_name, policy_constructor in policies.items():
        print(policy_name)
        for cache_size in range(200, 2000, 200):
            index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
            policy = policy_constructor(cache_size)
            cache_hits = 0
            for i_embed, embed in enumerate(embeds):
                policy_size = policy.count_items()
                assert(index.ntotal == policy_size)
                embed_as_array = embed.reshape(1, dim)
                distances, neighbors = index.search(embed_as_array, max(1,index.ntotal))
                if neighbors[0][0] != -1 and distances[0][0] <= same_embed_distance:
                    cache_hits += 1
                    neigh_embed = embeds[neighbors[0][0]]
                    id_remove, _ = policy.log_access(neighbors[0][0], neigh_embed, list(distances[0]))
                else:
                    id_remove, add_id = policy.log_access(i_embed, embed, list(distances[0]))
                    if add_id:
                        index.add_with_ids(embed_as_array, np.array([i_embed]))
                if id_remove != -1:
                    index.remove_ids(np.array([id_remove]))
            iter_results = {
                "Hit Ratio": cache_hits / len(embeds)
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