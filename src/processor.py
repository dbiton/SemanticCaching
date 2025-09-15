from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import os
import pickle
import random
import threading
import time
import uuid
import faiss
import numpy as np
import tqdm
import multiprocessing as mp

from caches.cache import (
    LFU,
    LRU,
    DistanceLFU,
    MissLFU,
    SphereQueryLFU,
    Surprisal,
    SurprisalLFU,
    DynamicAgingLFU,
    ClusterLFU,
    RAP,
    RR,
    FIFO,
    LIFO
)
from caches.arc import ARC
from caches.cluster_lru import ClusterLRU
from caches.lru_k import LRUK
from caches.OPT import OPT, ClusterOPT
from vector_stores.hnswlib_interface import HNSWVectorStore
from vector_stores.milvus_interface import MilvusVectorStore
from vector_stores.naive_interface import NaiveVectorStore

dataset_filenames = {
    "Quora": "datasets/embeds_quora_qp.pkl",
    "ELI5": "datasets/embeds_eli5.pkl",
    "NaturalQuestions": "datasets/embeds_nq.pkl",
    "MsMarco": "datasets/embeds_msmarco.pkl",
    "WildChat": "datasets/embeds_wildchat.pkl",
    "StackOverflow": "datasets/embeds_stackoverflow.pkl",
}


def get_flat_index_faiss(dim: int = 384):
    return faiss.IndexIDMap2(faiss.IndexFlatL2(dim))


def yield_batch_slices(total_size, batch_size):
    for start in range(0, total_size, batch_size):
        stop = min(start + batch_size, total_size)
        yield slice(start, stop), list(range(start, stop))


def load_embeds(dataset_name: str, N: int):
    path = dataset_filenames[dataset_name]
    with open(path, "rb") as f:
        data = pickle.load(f)
    embeds_texts = data["text"][:N]
    embeds = np.array(data["normalized_embeds"][:N], dtype=np.float32)
    return embeds, embeds_texts

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

def get_flat_index_milvus_lite(dim: int = 384):
    db_path = str(uuid.uuid4()).replace("-","").replace("_", "")+".db"
    return MilvusVectorStore(
        uri=db_path,
        collection_name="col",
        dim=dim,
        metric_type="L2",
        index_type="FLAT",
        index_params={},  # FLAT has no extra params
    )

def get_flat_index_milvus_standalone(uri: str = "http://localhost:19530", dim: int = 384):
    id = f"vector{uuid.uuid4()}".replace("-", "_")
    return MilvusVectorStore(
        uri=uri,
        collection_name=id,
        dim=dim,
        metric_type="L2",
        index_type="FLAT",
        index_params={},  # FLAT has no extra params
    )

def get_hnsw_index_milvus_standalone(uri: str = "http://localhost:19530", dim: int = 384):
    id = f"vector{uuid.uuid4()}".replace("-", "_")
    return MilvusVectorStore(
        uri=uri,
        collection_name=id,
        dim=dim,
        metric_type="L2",
        index_type="HNSW",
        index_params={},  # FLAT has no extra params
    )

def get_ivf_index_milvus_standalone(uri: str = "http://localhost:19530", dim: int = 384):
    id = f"vector{uuid.uuid4()}".replace("-", "_")
    return MilvusVectorStore(
        uri=uri,
        collection_name=id,
        dim=dim,
        metric_type="L2",
        index_type="IVF_FLAT",
        index_params={},  # FLAT has no extra params
    )

def get_hnsw_index_hnswlib(dim: int = 384):
    return HNSWVectorStore(
        dim=dim,
        M=16,
        ef_construction=200,
        allow_replace_delete=True
    )


def get_flat_index_naive(dim: int = 384):
    return NaiveVectorStore(dim=dim)


def get_ivf_index_milvus(
    uri: str = "http://localhost:19530", dim: int = 384, nlist: int = 10
):
    id = f"vector{uuid.uuid4()}".replace("-", "_")
    return MilvusVectorStore(
        uri=uri,
        collection_name=id,
        dim=dim,
        metric_type="L2",
        index_type="IVF_FLAT",
        index_params={"nlist": int(nlist)},
    )


def _run_single_worker(args):
    (
        dataset_name,
        cache_name,
        index_name,
        same_embed_distance,
        num_samples,
        cache_size,
        batch_size,
        count_nn,
        output_path,
        progress_queue,
    ) = args

    total_embeds, total_embeds_texts = load_embeds(dataset_name, num_samples)

    caches = {
        "MissLFU": (MissLFU, same_embed_distance),
        "NaiveRVB": (OPT, same_embed_distance, total_embeds),
        "ClusterRVB": (ClusterOPT, same_embed_distance, total_embeds),
        "SurprisalLFU": (SurprisalLFU, same_embed_distance),
        "Surprisal": (Surprisal, same_embed_distance),
        "LIFO": (LIFO, same_embed_distance),
        "FIFO": (FIFO, same_embed_distance),
        "RR": (RR, same_embed_distance),
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
    indices = {
        "milvus-lite": get_flat_index_milvus_lite,
        "milvus-standalone-flat": get_flat_index_milvus_standalone,
        "milvus-standalone-ivf": get_ivf_index_milvus_standalone,
        "milvus-standalone-hnsw": get_hnsw_index_milvus_standalone,
        "faiss": get_flat_index_faiss,
        "HotSwap": get_flat_index_naive,
        "hnswlib": get_hnsw_index_hnswlib
    }

    index = indices[index_name]()

    # Initialize cache
    cache_tuple = caches[cache_name]
    cache_constructor = cache_tuple[0]
    cache_args = cache_tuple[1:]
    cache = cache_constructor(*cache_args)
    cache.initialize(cache_size, index)

    runtime = 0

    # consts (simulated)
    llm_call_time = 100
    cache_access_time = 1

    # Stats
    total_hits = 0
    at_least_1_hits = 0
    simulated_runtime = 0
    total_hit_distance = 0
    
    for sl, i_embeds in yield_batch_slices(len(total_embeds), batch_size):
        simulated_runtime += cache_access_time
        embeds = total_embeds[sl]
        embeds_texts = total_embeds_texts[sl]
        _, cache_hits_dists = cache.get_cache_hits(embeds, count_nn)
        total_hit_distance += sum([sum(dists) for dists in cache_hits_dists])
        time_start = time.perf_counter()
        iter_cache_hits, _ = cache.cache(embeds, i_embeds, count_nn, embeds_texts)
        runtime += time.perf_counter() - time_start
        total_hits += np.sum(iter_cache_hits)
        reqs_hit = len(np.where(iter_cache_hits > 0)[0])
        at_least_1_hits += reqs_hit
        items_include = set(i_embeds).intersection(list(cache.items))
        items_miss = set(np.array(i_embeds)[np.where(iter_cache_hits == 0)[0]])
        items_llm = items_miss | items_include
        simulated_runtime += len(items_llm) * llm_call_time
        if progress_queue is not None:
            try:
                progress_queue.put(len(i_embeds))
            except (EOFError, BrokenPipeError):
                # Main process likely finished/cancelled; just continue quietly.
                pass

    fractional_recall_at_k = total_hits / (len(total_embeds) * count_nn)
    hit_rate = at_least_1_hits / len(total_embeds)
    mean_hit_distance = total_hit_distance / total_hits if total_hits > 0 else 0
    iter_results = {
        "Dataset": dataset_name,
        "Index": index_name,
        "Cache Name": cache_name,
        "Recall@K": fractional_recall_at_k,
        "Hit Rate": hit_rate,
        "Runtime": runtime,
        "Throughput": num_samples / runtime,
        "Cache Size": cache_size,
        "Same Embed Distance": same_embed_distance,
        "Simulated Runtime": simulated_runtime,
        "Output Path": output_path,
        "Batch Size": batch_size,
        "Count NN": count_nn,
        "Mean Hit Distance": mean_hit_distance
    }
    return iter_results


class Processor:
    def __init__(self, num_procs: int):
        self.num_procs = num_procs
        self.progress_queue = None
        self.progress_queue_thread = None
        self.stop_token = -1
        self.submissions = []
        self.num_batches = 0

    def setup_progress_queue(self):
        pbar = tqdm.tqdm(total=self.num_batches, desc="Running...")
        manager = mp.Manager()
        self.progress_queue = manager.Queue()

        def consumer(pbar):
            while True:
                item = self.progress_queue.get()
                if item == self.stop_token:
                    break
                pbar.update(item)

        self.progress_queue_thread = threading.Thread(
            target=consumer, args=(pbar,), daemon=True
        )
        self.progress_queue_thread.start()

    def submit(
        self,
        dataset_name: str,
        cache_name: str,
        index_name: str,
        same_embed_distance: float,
        dataset_size: int,
        cache_size: int,
        batch_size: int,
        count_nn: int,
        output_path: str,
    ) -> None:
        if os.path.exists(output_path):
            os.remove(output_path)
        self.num_batches += dataset_size
        self.submissions.append(
            [
                dataset_name,
                cache_name,
                index_name,
                same_embed_distance,
                dataset_size,
                cache_size,
                batch_size,
                count_nn,
                output_path,
            ]
        )

    def load_embeds(dataset_name: str, N: int):
        path = dataset_filenames[dataset_name]
        with open(path, "rb") as f:
            data = pickle.load(f)
        embeds_texts = data["text"][:N]
        embeds = np.array(data["normalized_embeds"][:N], dtype=np.float32)
        return embeds, embeds_texts

    def run(self) -> None:
        self.setup_progress_queue()
        Pool = ProcessPoolExecutor if self.num_procs > 1 else ThreadPoolExecutor
        random.shuffle(self.submissions)
        with Pool(self.num_procs) as executor:
            futures = [
                executor.submit(_run_single_worker, arg + [self.progress_queue])
                for arg in self.submissions
            ]
            for future in as_completed(futures):
                result = future.result()
                output_path = result["Output Path"]
                with open(output_path, "a") as f:
                    json.dump(result, f)
                    f.write("\n")
        self.progress_queue.put(self.stop_token)
        self.progress_queue_thread.join()
