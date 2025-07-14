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
import tqdm
from cache import *
from OPT import FreqOPT, RelaxedLearnedOPT, RelaxedOPT, OPT
from lrfu import LRFU, DeltaLRFU, HillClimbingLRFU
from reduce_dim import reduce_dim
from surprisal_reg_cache import SurprisalReg

dataset_filenames = {
    "persona": "datasets_text/embeds_persona.pkl",
    "quora": "datasets_text/embeds_quora.pkl",
    "OAsst": "datasets_text/embeds_oasst.pkl",
    "WildChat": "datasets_text/embeds_chat.pkl",
    "Bing": "datasets_text/embeds_bing.pkl",
    "StackOverflow": "datasets_text/embeds_so.pkl",
    "ComQA": "datasets_text/embeds_ComQA.pkl",
    # "Steam": "datasets/embeds_steam.pkl",
}

has_gpu = False
NUM_PROCS = 4

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
    for dataset_name, path in dataset_filenames.items():
        if not os.path.exists(path):
            print(f"Skipping \"{path}\" because it does not exist")
            continue
        with open(path, "rb") as f:
            embeds = pickle.load(f)
            yield dataset_name, embeds

def yield_batches(lst, k):
    for i in range(0, len(lst), k):
        yield lst[i:i + k], list(range(i, min(i+k, len(lst))))


def process(args):
    (pbar, cache, cache_size, dim, embeds, cache_name, batch_size, count_nn) = args
    index = faiss.IndexIDMap2(faiss.IndexFlatL2(dim))
    cache.initialize(cache_size, index)
    cache_hits = 0
    t0 = time.time()
    for batch_embeds, i_embeds in yield_batches(embeds, batch_size):
        embeds_texts = [v for (_, v) in batch_embeds]
        embeds_embeds = np.array([v for (v, _) in batch_embeds])
        iter_cache_hits, evicted_embeds_ids = cache.request(embeds_embeds, i_embeds, count_nn, embeds_texts)
        cache_hits += np.count_nonzero(iter_cache_hits)
        if pbar is not None: pbar.update(1)
    iter_results = {
        "Cache Name": cache_name,
        "Hit Ratio": cache_hits / len(embeds),
        "Runtime": time.time() - t0,
        "Cache Size": cache_size
    }
    return iter_results

def process_layered(args):
    (pbar, cache, cache_size, dim, embeds, cache_name, batch_size, count_nn) = args
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
    cache.initialize(cache_size, index)
    cache_hits = 0
    l2_cache_hits = 0
    t0 = time.time()
    index_unlimited = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
    for batch_embeds, i_embeds in yield_batches(embeds, batch_size):
        iter_cache_hits, evicted_embeds_ids = cache.request(batch_embeds, i_embeds, count_nn)
        iter_cache_hits_count = np.count_nonzero(iter_cache_hits)
        cache_hits += iter_cache_hits_count
        if iter_cache_hits_count != len(iter_cache_hits):
            i_embeds_cache_misses = np.where(iter_cache_hits == 0)[0] + min(i_embeds)
            batch_embeds_misses = embeds[i_embeds_cache_misses]
            distances_sqrd, neighbors = index_unlimited.search(batch_embeds_misses, 1)
            distances = np.sqrt(distances_sqrd)
            l2_cache_hits += np.sum(distances <= cache.same_embed_distance)
        if len(evicted_embeds_ids) > 0:
            evicted_embed = embeds[evicted_embeds_ids]
            index_unlimited.add_with_ids(evicted_embed, np.array(evicted_embeds_ids))
        if pbar is not None: pbar.update(1)
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
    batch_size = 1
    count_nn = 1
    num_samples = 15000
    MAX_CACHE_SIZE = 0.1
    COUNT_STEPS = 10
    dim = 384
    same_embed_distance = 0.5
    for dataset_name, data in load_embeds():
        print(f"loaded {dataset_name} with {len(data['embeds'])} examples...")
        indices = random.sample(range(len(data['embeds'])), min(num_samples, len(data['embeds'])))
        preps = [data['text'][i] for i in indices]
        embeds_actual = reduce_dim(np.array([data['embeds'][i] for i in indices]), dim)
        embeds = list(zip(embeds_actual, preps))
        print("loaded!")
        caches = {
            "Surprisal": Surprisal(same_embed_distance),
            # "HillClimbingLRFU": HillClimbingLRFU(same_embed_distance, .1, num_samples // 25),
            # "SurprisalReg": SurprisalReg(same_embed_distance),
            #"CountChars": CountChars(same_embed_distance),
            #"CountWords": CountWords(same_embed_distance),
            # "HillClimbingLRFU": HillClimbingLRFU(same_embed_distance),
            #"Freq": FreqOPT(same_embed_distance, dim=dim),
            "RL_OPT": RelaxedLearnedOPT(same_embed_distance, dim=dim),
            "R_OPT": RelaxedOPT(same_embed_distance, embeds_actual),
            "OPT": OPT(same_embed_distance, embeds_actual),
            #"BetterTinyLFU": BetterTinyLFU(same_embed_distance),
            #"TinyLFU": TinyLFU(same_embed_distance),
            #"RAP": RAP(same_embed_distance),
            #"LRFU.1": LRFU(same_embed_distance, .1),
            #"LRFU.01": LRFU(same_embed_distance, .01),
            #"LRFU1.": LRFU(same_embed_distance, 1),
            "LRU": LRU(same_embed_distance),
            #"PCA": PCA(same_embed_distance),
            #"Radius": FixedRadius(same_embed_distance, similar_embed_distance),
            #"Dummy": Dummy(same_embed_distance),
            #"RR": RR(same_embed_distance),
            #"DistanceLFU": DistanceLFU(same_embed_distance),
            "LFU": LFU(same_embed_distance),
            #"SphereLFU": SphereQueryLFU(same_embed_distance),
            #"DALFU": PeriodicAgingLFU(same_embed_distance, aging_interval=720, aging_factor=0.5),
        }

        results = {}
        args = []
        for cache_name, cache in caches.items():
            step_size = int(num_samples * MAX_CACHE_SIZE // COUNT_STEPS)
            for cache_size in range(step_size, int(num_samples * MAX_CACHE_SIZE), step_size):
                args.append((copy.deepcopy(cache), cache_size, dim, embeds, cache_name, batch_size, count_nn))
        num_batches = sum([len(embeds)/batch_size for _, _, _, embeds, _, batch_size, _ in args])
        if NUM_PROCS > 1:
            pbar = tqdm.tqdm(total=len(args), desc=f"Processing {dataset_name} with {num_batches} batches")
            args = [(None, *arg) for arg in args]
            with ProcessPoolExecutor(NUM_PROCS) as executor:
                raw_results = executor.map(process, args)
        else:
            pbar = tqdm.tqdm(total=num_batches, desc=f"Processing {dataset_name} with {num_batches} batches")
            args = [(pbar, *arg) for arg in args]
            raw_results = map(process, args)
            
        for iter_results in raw_results:
            if NUM_PROCS > 1: 
                pbar.update(1)
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
