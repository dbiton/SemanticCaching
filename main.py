from concurrent.futures import ProcessPoolExecutor
import copy
from itertools import chain
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
from cache_denstream import *
from cache_kmeans import CacheKMeans

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
        "Bing": "embeds_bing.pkl",
    }
    for dataset_name, filename in filenames.items():
        with open(filename, "rb") as f:
            embeds = pickle.load(f)
            yield dataset_name, embeds

def process(args):
    (cache, cache_size, dim, embeds, cache_name) = args
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
    cache.initialize(cache_size, index)
    cache_hits = 0
    t0 = time.time()
    for i_embed, embed in enumerate(embeds):
        cache_hit, evicted_embed_id = cache.request(embed.reshape(1, -1), i_embed)
        if cache_hit:
            cache_hits += 1
    iter_results = {
        "Cache Name": cache_name,
        "Hit Ratio": cache_hits / len(embeds),
        "Runtime": time.time() - t0,
        "Cache Size": cache_size
    }
    return iter_results

def process_layered(args):
    (cache, cache_size, dim, embeds, cache_name) = args
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
    cache.initialize(cache_size, index)
    cache_hits = 0
    l2_cache_hits = 0
    t0 = time.time()
    index_unlimited = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
    for i_embed, embed in enumerate(embeds):
        cache_hit, evicted_embed_id = cache.request(embed.reshape(1, -1), i_embed)
        if cache_hit:
            cache_hits += 1
        else:
            distances, neighbors = index_unlimited.search(embed.reshape(1, -1), 1)
            if distances[0][0] <= cache.same_embed_distance:
                l2_cache_hits += 1
        if evicted_embed_id:
            evicted_embed = embeds[evicted_embed_id]
            index_unlimited.add_with_ids(evicted_embed.reshape(1, -1), np.array([evicted_embed_id]))
    # once something is evicted from the small cache, we put it into the larger unlimited cache
    iter_results = {
        "Cache Name": cache_name,
        "Hit Ratio L1": cache_hits / len(embeds),
        "Hit Ratio L2": l2_cache_hits / len(embeds),
        "Runtime Layered": time.time() - t0,
        "Cache Size": cache_size
    }
    return iter_results

def main():
    num_samples = 4000
    for dataset_name, embeds in load_embeds():
        print(f"loaded {dataset_name}...")
        sampled_indices = np.random.choice(embeds.shape[0], size=num_samples, replace=False)
        embeds = embeds[sampled_indices]
        print("loaded!")
        dim = 384
        same_embed_distance = 0.5
        similar_embed_distance = 1.0
        
        caches = {
            #"OPT2": OPT2(same_embed_distance, embeds),
            "Dummy": Dummy(same_embed_distance),
            #"OPT": OPT(same_embed_distance, embeds),
            #"ProximityScore": ProximityScore(1, 0.5),
            #"ProbMisses": ProbMisses,
            #"ProbMinCounter": ProbMinCounter,
            #"ProbMinDensity": ProbMinDensity,
            #"MinCounter": MinCounter(similar_embed_distance / 384),
            #"MinDensity": MinDensity,
            #"MaxDensity": MaxDensity,
            "RR": RR(same_embed_distance),
            "RAP": RAP(same_embed_distance),
            "LRU": LRU(same_embed_distance),
            #"KM": CacheKMeans(same_embed_distance, 10),
            "LFU": LFU(same_embed_distance),
            #"DS": DenStreamCache(same_embed_distance),
            # "FIFO": FIFO,
        }
        
        results = {}
        
        with ProcessPoolExecutor(16) as executor:
            args = []
            for cache_name, cache in caches.items():
                for cache_size in range(num_samples // 20, int(num_samples // 2), int(num_samples // 20)):
                    args.append((copy.deepcopy(cache), cache_size, dim, embeds, cache_name))
            raw_results = chain(executor.map(process_layered, args), executor.map(process, args))
            # raw_results = [process(arg) for arg in args]

        for iter_results in raw_results:
            for prop_name, prop in iter_results.items():
                cache_name = iter_results['Cache Name']
                cache_size = iter_results['Cache Size']
                if prop_name == 'Cache Name':
                    continue
                if prop_name not in results:
                    results[prop_name] = {}
                if cache_name not in results[prop_name]:
                    results[prop_name][cache_name] = {}
                results[prop_name][cache_name][cache_size] = prop
            print(cache_name, iter_results)
        plot(dataset_name, results)

if __name__=="__main__":
    main()