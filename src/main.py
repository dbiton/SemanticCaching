import sys
from concurrent.futures._base import as_completed
from typing import List, Dict
from concurrent.futures.thread import ThreadPoolExecutor
from util.faiss_hnsw_delete_wrapper import DeletesOnlyWrapper
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
from util.reduce_dim import reduce_dim
from util.faiss_like_hnsw import FaissLikeHNSW

dataset_filenames = {
    "WildChat": "datasets/embeds_wildchat.pkl",
    "Quora": "datasets/embeds_quora_qp.pkl",
    "StackOverflow": "datasets/embeds_stackoverflow.pkl",
    "ELI5": "datasets/embeds_eli5.pkl",
    "NaturalQuestions": "datasets/embeds_nq.pkl",
    "MsMarco": "datasets/embeds_msmarco.pkl",
}

NUM_PROCS = 1

def get_metrics(embeds: List[np.ndarray], texts: List[str]) -> Dict[str, float]:
    metrics = {}
    metrics['size'] = len(embeds)
    return metrics

def plot(dataset_name, results):
    for prop_name, prop_results in results.items():
        plt.figure()
        i = 0
        linestyles = ["-", "--", "-.", ":"]
        markers = ["o", "s", "^", "v", "D", "<", ">", "X", "+"]
        for cache_name, prop in prop_results.items():
            cache_size = list(prop.keys())
            prop_values = list(prop.values())
            plt.plot(cache_size, prop_values, label=cache_name, marker=markers[i % len(linestyles)], linestyle=linestyles[i % len(linestyles)])
            i += 1
        plt.xlabel("Cache Size")
        plt.ylabel(prop_name)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        figures_dir = "figures"
        plt.savefig(os.path.join(figures_dir, f"{prop_name}_{dataset_name}.png"))

def load_embeds():
    for dataset_name, path in dataset_filenames.items():
        if not os.path.exists(path):
            print(f"Skipping \"{path}\" because it does not exist")
            continue
        with open(path, "rb") as f:
            embeds = pickle.load(f)
            yield dataset_name, embeds

def yield_batches_indices(total_size, batch_size):
    for i in range(0, total_size, batch_size):
        yield list(range(i, min(i+batch_size, total_size)))

import numpy as np
import time
import faiss


def process(args):
    (
        index_name,
        index_constr,
        cache_tuple,          # (constructor, *args)
        cache_size,
        dim,
        total_embeds,
        total_embeds_texts,
        cache_name,
        batch_size,
        count_nn,
    ) = args

    index = index_constr()
    assert batch_size == 1  # code assumes one query per batch

    # Initialize cache
    cache_constructor = cache_tuple[0]
    cache_args = cache_tuple[1:]
    cache = cache_constructor(*cache_args)
    cache.initialize(cache_size, index)

    # Runtime bookkeeping
    t0 = time.time()
    runtime_per_l1 = 1
    runtime_per_l2 = 10
    runtime_per_llm = 100
    runtime = 0

    # Stats
    total_hits = 0
    at_least_1_hits = 0
    
    # Unlimited fallback index
    index_unlimited = faiss.IndexIDMap2(faiss.IndexFlatL2(dim))

    for i_embeds in yield_batches_indices(len(total_embeds), batch_size):
        embeds = total_embeds[i_embeds]
        embeds_texts = total_embeds_texts[i_embeds]

        # Cache lookup
        iter_cache_hits, evicted_embeds_ids = cache.request(embeds, i_embeds, count_nn, embeds_texts)
        # iter_cache_hits: shape (batch_size,), each entry ∈ [0, count_nn]

        runtime += runtime_per_l1
        total_hits += np.sum(iter_cache_hits)
        at_least_1_hits += np.any(iter_cache_hits)

        # L2/LLM fallback for total misses
        if np.any(iter_cache_hits == 0):
            runtime += runtime_per_l2

            miss_indices = np.array(i_embeds)[iter_cache_hits == 0]
            batch_embeds_misses = total_embeds[miss_indices]

            distances_sqrd, _ = index_unlimited.search(batch_embeds_misses, 1)
            distances = np.sqrt(distances_sqrd)
            count_found = np.count_nonzero(distances <= cache.same_embed_distance)
            count_missing = len(batch_embeds_misses) - count_found
            runtime += runtime_per_llm * count_missing

        # Update fallback index
        if len(evicted_embeds_ids) > 0:
            evicted_vectors = total_embeds[evicted_embeds_ids]
            index_unlimited.add_with_ids(evicted_vectors, np.array(evicted_embeds_ids))

    fractional_recall_at_k = total_hits / (len(total_embeds) * count_nn)
    binary_recall_at_k = at_least_1_hits / len(total_embeds)

    iter_results = {
        "Index": index_name,
        "Sim. Runtime": runtime,
        "Cache Name": cache_name,
        "Recall@K": fractional_recall_at_k,
        "AtLeast1@K": binary_recall_at_k,
        "Runtime": time.time() - t0,
        "Cache Size": cache_size
    }

    return iter_results


def get_hnsw_index():
    d, M = 384, 32
    base = faiss.IndexHNSWFlat(d, M)
    idmap = faiss.IndexIDMap2(base)
    hnsw_index = DeletesOnlyWrapper(idmap)
    return hnsw_index

def get_flat_index():
    dim = 384
    return faiss.IndexIDMap2(faiss.IndexFlatL2(dim))

def main():
    batch_size = 1
    count_nn = 1
    num_samples = 10000
    MAX_CACHE_SIZE = 0.25
    COUNT_STEPS = 5
    dim = 384
    same_embed_distance = .75
    for dataset_name, data in load_embeds():
        print(f"loaded {dataset_name} with {len(data['embeds'])} examples...")
        indices = list(range(num_samples))
        embeds_texts = np.array([data['text'][i] for i in indices])
        embeds = reduce_dim(np.array([data['embeds'][i] for i in indices]), dim).astype(np.float32)
        print("loaded!")
        caches = {
            #"NaiveRVB": (OPT, same_embed_distance, embeds),
            #"ClusterRVB": (ClusterOPT, same_embed_distance, embeds),
            #"SurprisalLFU": (SurprisalLFU, same_embed_distance),
            #"Surprisal": (Surprisal, same_embed_distance),
            "LFU": (LFU, same_embed_distance),
            #"LRU": (LRU, same_embed_distance),
            #"LRUK": (LRUK, same_embed_distance, 2),            
            #"DALFU": (DynamicAgingLFU, same_embed_distance, 32),
            #"ARC": (ARC, same_embed_distance),
            #"ClusterLRU": (ClusterLRU, same_embed_distance),
            #"ClusterLFU": (ClusterLFU, same_embed_distance),
            #"DistanceLFU": (DistanceLFU, same_embed_distance),
            #"RAP": (RAP, same_embed_distance),
            #"SphereLFU": (SphereQueryLFU, same_embed_distance),
        }

        # IVF needs a coarse quantizer
        nlist = 10  # number of Voronoi cells
        coarse_quantizer = faiss.IndexFlatL2(dim)
        ivf_index = faiss.IndexIVFFlat(coarse_quantizer, dim, nlist)
        train_embeds = embeds[:5000]
        ivf_index.train(train_embeds)
        
        faiss_indices = {
            "hnsw": get_hnsw_index,
            #"ivf": faiss.IndexIDMap2(ivf_index),
            "flat": get_flat_index,
        }
        
        args = []
        for cache_name, create_cache_args in caches.items():
            step_size = int(num_samples * MAX_CACHE_SIZE // COUNT_STEPS)
            for faiss_index_name, faiss_index in faiss_indices.items():
                for cache_size in range(step_size, int(num_samples * MAX_CACHE_SIZE), step_size):
                    args.append((faiss_index_name, faiss_index, create_cache_args, cache_size, dim, embeds, embeds_texts, cache_name, batch_size, count_nn))
        
        num_batches = len(caches) * COUNT_STEPS * len(faiss_indices)
        pbar = tqdm.tqdm(total=num_batches, desc=f"Processing {dataset_name} with {num_batches} batches")
        results = {}
        Pool = ProcessPoolExecutor if NUM_PROCS > 1 else ThreadPoolExecutor
        with Pool(NUM_PROCS) as executor:
            futures = [executor.submit(process, arg) for arg in args]
            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)
                for prop_name, prop in result.items():
                    _cache_name = result['Cache Name']
                    cache_index = result['Index']
                    cache_name = f"{_cache_name}_{cache_index}"
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
