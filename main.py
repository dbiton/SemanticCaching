from concurrent.futures import ProcessPoolExecutor
import copy
from itertools import chain
import json
import os
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
run_parallel = True

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

def embed_titles(n=-1):
    with open("Titles.txt", encoding="utf-8") as f:
        titles_strings = f.readlines(n)
    titles_embeddings = [e.tolist() for e in embed_strings(titles_strings)]
    with open("TitlesEmbeddings.json", "w") as f:
        json.dump(list(zip(titles_strings, titles_embeddings)), f)

def plot(dataset_name, results):
    for prop_name, prop_results in results.items():
        plt.figure()
        i = 0
        markers = ["o", "s", "^", "v", "D", "<", ">", "X", "+"]
        for cache_name, prop in prop_results.items():
            cache_size = list(prop.keys())
            prop_values = list(prop.values())
            plt.plot(cache_size, prop_values, label=cache_name, marker=markers[i])
            i += 1
        plt.xlabel("Cache Size")
        plt.ylabel(prop_name)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        figures_dir = "figures"
        plt.savefig(os.path.join(figures_dir, f"{dataset_name}_{prop_name}.png"))

def generate_embeds(mean, std_dev, dim, length):    
    np.random.seed(0)
    return [np.random.normal(mean, std_dev, dim) for _ in range(length)]

def load_embeds():
    filenames = {
        "Bing": "embeds_bing.pkl",
        "StackOverflow": "embeds_so.pkl",
        "WildChat": "embeds_chat.pkl"
    }
    embeds_dir = "datasets"
    for dataset_name, filename in filenames.items():
        with open(os.path.join(embeds_dir, filename), "rb") as f:
            embeds = pickle.load(f)
            yield dataset_name, embeds

def yield_batches(lst, k):
    for i in range(0, len(lst), k):
        yield lst[i:i + k], list(range(i, min(i+k, len(lst))))


def process(args):
    (cache, cache_size, dim, embeds, cache_name, batch_size, count_nn) = args
    index = faiss.IndexIDMap2(faiss.IndexFlatL2(dim))
    cache.initialize(cache_size, index)
    cache_hits = 0
    t0 = time.time()
    for batch_embeds, i_embeds in yield_batches(embeds, batch_size):
        iter_cache_hits, evicted_embeds_ids = cache.request(batch_embeds, i_embeds, count_nn)
        cache_hits += np.sum(np.any(iter_cache_hits, axis=1))
    iter_results = {
        "Cache Name": cache_name,
        "Hit Ratio": cache_hits / len(embeds),
        "Runtime": time.time() - t0,
        "Cache Size": cache_size
    }
    return iter_results

def process_layered(args):
    (cache, cache_size, dim, embeds, cache_name, batch_size, count_nn) = args
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
    cache.initialize(cache_size, index)
    cache_hits = 0
    l2_cache_hits = 0
    t0 = time.time()
    index_unlimited = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
    for batch_embeds, i_embeds in yield_batches(embeds, batch_size):
        iter_cache_hits, evicted_embeds_ids = cache.request(batch_embeds, i_embeds, count_nn)
        iter_cache_hits_count = np.sum(np.any(iter_cache_hits, axis=1))
        cache_hits += iter_cache_hits_count
        if iter_cache_hits_count != len(iter_cache_hits):
            i_embeds_cache_misses = np.where(np.any(iter_cache_hits, axis=1) != True)[0] + min(i_embeds)
            batch_embeds_misses = embeds[i_embeds_cache_misses]
            distances_sqrd, neighbors = index_unlimited.search(batch_embeds_misses, 1)
            distances = np.sqrt(distances_sqrd)
            l2_cache_hits += np.sum(distances <= cache.same_embed_distance)
        if len(evicted_embeds_ids) > 0:
            evicted_embed = embeds[evicted_embeds_ids]
            index_unlimited.add_with_ids(evicted_embed, np.array(evicted_embeds_ids))
    iter_results = {
        "Cache Name": cache_name,
        "Hit Ratio L1+L2": (cache_hits + l2_cache_hits) / len(embeds),
        "Hit Ratio L1": cache_hits / len(embeds),
        "Hit Ratio L2": l2_cache_hits / len(embeds),
        "Runtime Layered": time.time() - t0,
        "Cache Size": cache_size
    }
    return iter_results

def main():
    batch_size = 10
    count_nn = 1
    num_samples = 10000
    for dataset_name, embeds in load_embeds():
        print(f"loaded {dataset_name}...")
        embeds = embeds[:num_samples]
        print("loaded!")
        dim = 384
        same_embed_distance = 0.75
        similar_embed_distance = 1.0

        caches = {
            "OPT": OPT(same_embed_distance, embeds),
            #"PCA": PCA(same_embed_distance),
            "Radius": FixedRadius(same_embed_distance, similar_embed_distance),
            "Dummy": Dummy(same_embed_distance),
            #"RR": RR(same_embed_distance),
            "RAP": RAP(same_embed_distance),
            "LRU": LRU(same_embed_distance),
            "LFU": LFU(same_embed_distance)
        }

        results = {}
        args = []
        for cache_name, cache in caches.items():
            for cache_size in range(num_samples // 20, int(num_samples) // 2, int(num_samples // 20)):
                args.append((copy.deepcopy(cache), cache_size, dim, embeds, cache_name, batch_size, count_nn))

        if run_parallel:
            with ProcessPoolExecutor(16) as executor:
                raw_results = chain(executor.map(process_layered, args), executor.map(process, args))
        else:
            raw_results = chain(map(process_layered, args), map(process, args))

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
