import sys
from concurrent.futures._base import as_completed
import threading
from typing import List, Dict
from concurrent.futures.thread import ThreadPoolExecutor
import uuid

from vector_stores.naive_interface import NaiveVectorStore
sys.path.append(".")

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import os
import pickle
from random import sample
import time
from matplotlib import pyplot as plt
import numpy as np
import tqdm
import numpy as np
import time
import faiss

from caches.cache import *
from caches.arc import ARC
from caches.cluster_lru import ClusterLRU
from caches.lru_k import LRUK
from caches.OPT import OPT, ClusterOPT
from vector_stores.milvus_interface import MilvusVectorStore
from vector_stores.hnswlib_interface import HNSWVectorStore


dataset_filenames = {
    "Quora": "datasets/embeds_quora_qp.pkl",
    "ELI5": "datasets/embeds_eli5.pkl",
    "NaturalQuestions": "datasets/embeds_nq.pkl",
    "MsMarco": "datasets/embeds_msmarco.pkl",
    "WildChat": "datasets/embeds_wildchat.pkl",
    #"StackOverflow": "datasets/embeds_stackoverflow.pkl",
}

NUM_PROCS = 4

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

def get_embeds_paths():
    return list(dataset_filenames.items())

def load_embeds(path: str, N: int):
    with open(path, "rb") as f:
        data = pickle.load(f)
    embeds_texts = data['text'][:N]
    embeds = np.array(data['normalized_embeds'][:N], dtype=np.float32)
    return embeds, embeds_texts

def yield_batch_slices(total_size, batch_size):
    for start in range(0, total_size, batch_size):
        stop = min(start + batch_size, total_size)
        yield slice(start, stop), list(range(start, stop))

def process(args):
    (
        index_name,
        index_constr,
        cache_tuple,          # (constructor, *args)
        stream_size,
        cache_size,
        dim,
        dataset_path,
        cache_name,
        batch_size,
        count_nn,
        progress_queue,
    ) = args

    total_embeds, total_embeds_texts = load_embeds(dataset_path, stream_size)
    index = index_constr()

    # Initialize cache
    cache_constructor = cache_tuple[0]
    cache_args = cache_tuple[1:]
    cache = cache_constructor(*cache_args)
    cache.initialize(cache_size, index)

    t0 = time.time()

    # Stats
    total_hits = 0
    at_least_1_hits = 0
    
    for sl, i_embeds in yield_batch_slices(len(total_embeds), batch_size):
        embeds = total_embeds[sl]
        embeds_texts = total_embeds_texts[sl]
        iter_cache_hits, _ = cache.request(embeds, i_embeds, count_nn, embeds_texts)
        total_hits += np.sum(iter_cache_hits)
        at_least_1_hits += len(np.where(iter_cache_hits > 0)[0])
        if progress_queue is not None:
            progress_queue.put(len(i_embeds))

    fractional_recall_at_k = total_hits / (len(total_embeds) * count_nn)
    binary_recall_at_k = at_least_1_hits / len(total_embeds)

    iter_results = {
        "Index": index_name,
        "Cache Name": cache_name,
        "Recall@K": fractional_recall_at_k,
        "AtLeast1@K": binary_recall_at_k,
        "Runtime": time.time() - t0,
        "Cache Size": cache_size
    }
    return iter_results

def get_flat_index_faiss(dim: int = 384):
    return faiss.IndexIDMap2(faiss.IndexFlatL2(dim))

def get_hnsw_index_milvus(uri: str = "http://localhost:19530", dim: int = 384):
    id = f"vector{uuid.uuid4()}".replace("-", "_")
    return MilvusVectorStore(
        uri=uri,
        collection_name=id,
        dim=dim,
        metric_type="L2",
        index_type="HNSW",
        index_params={"M": 32, "efConstruction": 200},
    )

def get_flat_index_milvus(uri: str = "http://localhost:19530", dim: int = 384):
    id = f"vector{uuid.uuid4()}".replace("-", "_")
    return MilvusVectorStore(
        uri=uri,
        collection_name=id,
        dim=dim,
        metric_type="L2",
        index_type="FLAT",
        index_params={},           # FLAT has no extra params
    )

def get_hnsw_index_hnswlib(dim: int = 384):
    return HNSWVectorStore(
        dim = dim
    )

def get_flat_index_naive(dim: int = 384):
    return NaiveVectorStore(
        dim = dim
    )

def get_ivf_index_milvus(uri: str = "http://localhost:19530", dim: int = 384, nlist: int = 10):
    id = f"vector{uuid.uuid4()}".replace("-", "_")
    return MilvusVectorStore(
        uri=uri,
        collection_name=id,
        dim=dim,
        metric_type="L2",
        index_type="IVF_FLAT",
        index_params={"nlist": int(nlist)},
    )

def main():
    batch_size = 1
    count_nn = 1
    num_samples = 1000
    MAX_CACHE_SIZE = 0.1
    COUNT_STEPS = 10
    dim = 384
    same_embed_distance = .75
    for dataset_name, dataset_path in get_embeds_paths():
        print(f"processing {dataset_name}")
        # indices = list(range(num_samples))
        # embeds_texts = np.array([data['text'][i] for i in indices])
        # embeds = reduce_dim(np.array([data['embeds'][i] for i in indices]), dim).astype(np.float32)
        
        print("loaded!")
        caches = {
            #"NaiveRVB": (OPT, same_embed_distance, embeds),
            #"ClusterRVB": (ClusterOPT, same_embed_distance, embeds),
            #"SurprisalLFU": (SurprisalLFU, same_embed_distance),
            #"Surprisal": (Surprisal, same_embed_distance),
            "LFU": (LFU, same_embed_distance),
            "LRU": (LRU, same_embed_distance),
            "LRUK": (LRUK, same_embed_distance, 2),            
            "DALFU": (DynamicAgingLFU, same_embed_distance, 32),
            "ARC": (ARC, same_embed_distance),
            "ClusterLRU": (ClusterLRU, same_embed_distance),
            "ClusterLFU": (ClusterLFU, same_embed_distance),
            "DistanceLFU": (DistanceLFU, same_embed_distance),
            "RAP": (RAP, same_embed_distance),
            "SphereLFU": (SphereQueryLFU, same_embed_distance),
        }
        
        faiss_indices = {
            #"hnsw": get_hnsw_index,
            #"ivf": get_ivf_index,
            #"flat-milvus": get_flat_index_milvus,
            #"flat-faiss": get_flat_index_faiss,
            # "hnsw-hnswlib": get_hnsw_index_hnswlib,
            "flat-naive": get_flat_index_naive,
        }
        
        # feedback
        num_batches = len(caches) * len(faiss_indices) * num_samples * COUNT_STEPS
        pbar = tqdm.tqdm(total=num_batches, desc=f"Processing {dataset_name}'s {num_batches} batches")
        manager = mp.Manager()
        progress_queue = manager.Queue()
        stop_token = -1
        def consumer(pbar):
            while True:
                item = progress_queue.get()
                if item == -1:
                    break
                pbar.update(item)
        t = threading.Thread(target=consumer, args=(pbar,), daemon=True)
        t.start()
        # feedback
        
        args = []
        for cache_name, create_cache_args in caches.items():
            step_size = int(num_samples * MAX_CACHE_SIZE // COUNT_STEPS)
            for faiss_index_name, faiss_index in faiss_indices.items():
                for cache_size in range(step_size, int(num_samples * MAX_CACHE_SIZE), step_size):
                    args.append((faiss_index_name, faiss_index, create_cache_args, num_samples, cache_size, dim, dataset_path, cache_name, batch_size, count_nn, progress_queue))
        results = {}
        Pool = ProcessPoolExecutor if NUM_PROCS > 1 else ThreadPoolExecutor
        with Pool(NUM_PROCS) as executor:
            futures = [executor.submit(process, arg) for arg in args]
            for future in as_completed(futures):
                result = future.result()
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
        progress_queue.put(stop_token)
        t.join()
        plot(dataset_name, results)

if __name__=="__main__":
    main()
