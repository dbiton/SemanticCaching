import sys
from concurrent.futures._base import as_completed
sys.path.append(".")

from concurrent.futures import ProcessPoolExecutor
import copy
import os
import pickle
from random import sample
import time
import faiss
from matplotlib import pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
import tqdm

from caches.cache import *
from caches.arc import ARC
from caches.cluster_lru import ClusterLRU
from caches.lru_k import LRUK
from caches.OPT import RelaxedLearnedOPT, RelaxedOPT, OPT, ClusterOPT,\
    ClusterRelaxedOPT
from caches.lrfu import LRFU, DeltaLRFU, HillClimbingLRFU
from src.util.reduce_dim import reduce_dim

dataset_filenames = {
    "WildChat": "datasets/embeds_chat.pkl",
    "Bing": "datasets/embeds_bing.pkl",
    "StackOverflow": "datasets/embeds_so.pkl",
    "ComQA": "datasets/embeds_ComQA.pkl",
    "persona": "datasets/embeds_persona.pkl",
    "quora": "datasets/embeds_quora.pkl",
    "OAsst": "datasets/embeds_oasst.pkl",
    # "Steam": "datasets/embeds_steam.pkl",
}

has_gpu = False
NUM_PROCS = 4

def plot(dataset_name, results):
    for prop_name, prop_results in results.items():
        plt.figure()
        i = 0
        linestyles = ["-", "--", "-.", ":"]
        markers = ["o", "s", "^", "v", "D", "<", ">", "X", "+"]
        for cache_name, prop in prop_results.items():
            cache_size = list(prop.keys())
            prop_values = list(prop.values())
            plt.plot(cache_size, prop_values, label=cache_name, marker=markers[i], linestyle=linestyles[i % len(linestyles)])
            i += 1
        plt.xlabel("Cache Size")
        plt.ylabel(prop_name)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        figures_dir = "figures"
        plt.savefig(os.path.join(figures_dir, f"{dataset_name}_{prop_name}.png"))

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
    (cache_tuple, cache_size, dim, embeds, cache_name, batch_size, count_nn) = args
    index = faiss.IndexIDMap2(faiss.IndexFlatL2(dim))
    cache_constructor = cache_tuple[0]
    cache_args = cache_tuple[1:]
    cache = cache_constructor(*cache_args)
    cache.initialize(cache_size, index)
    cache_hits = 0
    t0 = time.time()
    for batch_embeds, i_embeds in yield_batches(embeds, batch_size):
        embeds_texts = [v for (_, v) in batch_embeds]
        embeds_embeds = np.array([v for (v, _) in batch_embeds])
        iter_cache_hits, evicted_embeds_ids = cache.request(embeds_embeds, i_embeds, count_nn, embeds_texts)
        cache_hits += np.count_nonzero(iter_cache_hits)
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
    num_samples = 10000
    MAX_CACHE_SIZE = 0.1
    COUNT_STEPS = 10
    dim = 384
    same_embed_distance = .75
    for dataset_name, data in load_embeds():
        print(f"loaded {dataset_name} with {len(data['embeds'])} examples...")
        indices = list(range(num_samples))
        preps = [data['text'][i] for i in indices]
        embeds_actual = reduce_dim(np.array([data['embeds'][i] for i in indices]), dim)
        embeds = list(zip(embeds_actual, preps))
        print("loaded!")
        caches = {
            #"ClusterOPT": ClusterOPT(same_embed_distance, embeds_actual),
            
            "ARC": (ARC, same_embed_distance),
            "LRU": (LRU, same_embed_distance),
            "LRUK": (LRUK, same_embed_distance, 2),
            "ClusterLRU": (ClusterLRU, same_embed_distance),
            "ClusterLFU": (ClusterLFU, same_embed_distance),
            "DistanceLFU": (DistanceLFU, same_embed_distance),
            "LFU": (LFU, same_embed_distance),
            "DALFU": (DynamicAgingLFU, same_embed_distance, 32),
            
            #"PCA": PCA(same_embed_distance),
            #"Radius": FixedRadius(same_embed_distance, similar_embed_distance),
            #"Dummy": Dummy(same_embed_distance),
            #"RR": RR(same_embed_distance),
            #"RAP": RAP(same_embed_distance),
            #"SphereLFU": SphereQueryLFU(same_embed_distance),

        }

        args = []
        for cache_name, create_cache_args in caches.items():
            step_size = int(num_samples * MAX_CACHE_SIZE // COUNT_STEPS)
            for cache_size in range(step_size, int(num_samples * MAX_CACHE_SIZE), step_size):
                args.append((create_cache_args, cache_size, dim, embeds, cache_name, batch_size, count_nn))
        
        num_batches = len(caches) * COUNT_STEPS
        pbar = tqdm.tqdm(total=num_batches, desc=f"Processing {dataset_name} with {num_batches} batches")
        results = {}
        with ProcessPoolExecutor(NUM_PROCS) as executor:
            futures = [executor.submit(process, arg) for arg in args]
            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)
                for prop_name, prop in result.items():
                    cache_name = result['Cache Name']
                    cache_size = result['Cache Size']
                    if prop_name == 'Cache Name':
                        continue
                    if prop_name not in results:
                        results[prop_name] = {}
                    if cache_name not in results[prop_name]:
                        results[prop_name][cache_name] = {}
                    results[prop_name][cache_name][cache_size] = prop
                print(cache_name, result)
        plot(dataset_name, results)

if __name__=="__main__":
    main()
