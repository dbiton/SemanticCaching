import json
import pickle
from random import sample
import time
from typing import Dict, List
import faiss
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from cache import *

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
    
def plot(dataset_name, results):
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
        plt.savefig(f"{dataset_name}_{prop_name}.png")

def generate_embeds(mean, std_dev, dim, len):    
    np.random.seed(0)
    return [np.random.normal(mean, std_dev, dim) for _ in range(len)]

def load_embeds():
    filenames = {
        "StackOverflow": "embeds_so.pkl",
        "Bing": "embeds_bing.pkl",
        "WildChat": "embeds_chat.pkl"
    }
    for dataset_name, filename in filenames.items():
        with open(filename, "rb") as f:
            embeds = pickle.load(f)
            yield dataset_name, embeds

def main():
    num_samples = 10000
    for dataset_name, embeds in load_embeds():
        print(f"loaded {dataset_name}...")
        sampled_indices = np.random.choice(embeds.shape[0], size=num_samples, replace=False)
        embeds = embeds[sampled_indices]
        print("loaded!")
        dim = 384
        same_embed_distance = 0.707
        similar_embed_distance = 1.0
        
        caches = {
            # "OPT": OPT(same_embed_distance, embeds),
            # "ProximityScore": ProximityScore(1, 0.5),
            #"ProbMisses": ProbMisses,
            #"ProbMinCounter": ProbMinCounter,
            #"ProbMinDensity": ProbMinDensity,
            # "MinCounter": MinCounter(similar_embed_distance / 384),
            #"MinDensity": MinDensity,
            #"MaxDensity": MaxDensity,
            #"RR": RR,
            "LRU": LRU(same_embed_distance),
            "LFU": LFU(same_embed_distance),
            # "FIFO": FIFO,
        }
        
        results = {}
        
        for cache_name, cache in caches.items():
            print(cache_name)
            for cache_size in range(num_samples // 10, num_samples, num_samples // 10):
                index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
                cache.initialize(cache_size, index)
                cache_hits = 0
                t0 = time.time()
                for i_embed, embed in enumerate(embeds):
                    if cache.request(embed.reshape(1, -1), i_embed):
                        cache_hits += 1
                iter_results = {
                    "Hit Ratio": cache_hits / len(embeds),
                    "Runtime": time.time() - t0
                }
                for prop_name, prop in iter_results.items():
                    if prop_name not in results:
                        results[prop_name] = {}
                    if cache_name not in results[prop_name]:
                        results[prop_name][cache_name] = {}
                    results[prop_name][cache_name][cache_size] = prop
                print(cache_name, iter_results)
        plot(dataset_name, results)

if __name__=="__main__":
    main()