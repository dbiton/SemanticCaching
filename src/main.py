import json
import sys
from concurrent.futures._base import as_completed
import threading
from typing import List, Dict
from concurrent.futures.thread import ThreadPoolExecutor
import uuid

import pandas as pd

from vector_stores.naive_interface import NaiveVectorStore

sys.path.append(".")

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import os
import pickle
import time
from matplotlib import pyplot as plt
import numpy as np
import tqdm
import faiss
import seaborn as sns

from caches.cache import (
    LFU,
    LRU,
    DistanceLFU,
    SphereQueryLFU,
    Surprisal,
    SurprisalLFU,
    DynamicAgingLFU,
    ClusterLFU,
    RAP,
)
from caches.arc import ARC
from caches.cluster_lru import ClusterLRU
from caches.lru_k import LRUK
from caches.OPT import OPT, ClusterOPT
from vector_stores.milvus_interface import MilvusVectorStore
from vector_stores.hnswlib_interface import HNSWVectorStore
from processor import Processor

# use only text, embeds - remove normalized embeds
dataset_filenames = {
    "Quora": "datasets/embeds_quora_qp.pkl",
    "ELI5": "datasets/embeds_eli5.pkl",
    "NaturalQuestions": "datasets/embeds_nq.pkl",
    "MsMarco": "datasets/embeds_msmarco.pkl",
    "WildChat": "datasets/embeds_wildchat.pkl",
    "StackOverflow": "datasets/embeds_stackoverflow.pkl",
}

NUM_PROCS = 8


def plot(dataset_name, results):
    for prop_name, prop_results in results.items():
        plt.figure()
        i = 0
        linestyles = ["-", "--", "-.", ":"]
        markers = ["o", "s", "^", "v", "D", "<", ">", "X", "+"]
        for cache_name, prop in prop_results.items():
            cache_size = list(prop.keys())
            prop_values = list(prop.values())
            plt.plot(
                cache_size,
                prop_values,
                label=cache_name,
                marker=markers[i % len(linestyles)],
                linestyle=linestyles[i % len(linestyles)],
            )
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


def load_embeds(dataset_name: str, N: int):
    path = dataset_filenames[dataset_name]
    with open(path, "rb") as f:
        data = pickle.load(f)
    embeds_texts = data["text"][:N]
    embeds = np.array(data["normalized_embeds"][:N], dtype=np.float32)
    return embeds, embeds_texts

def main():
    batch_size = 1
    count_nn = 1
    num_samples = 1000
    MAX_CACHE_SIZE = 0.1
    COUNT_STEPS = 10
    dim = 384
    same_embed_distance = 0.75

    faiss_indices_names = {"flat-naive"}
    caches_names = {"NaiveRVB", "ClusterRVB"}

    for dataset_name, _ in get_embeds_paths():
        print(f"processing {dataset_name}")

        args = []
        for cache_name in caches_names:
            step_size = int(num_samples * MAX_CACHE_SIZE // COUNT_STEPS)
            for index_name in faiss_indices_names:
                for cache_size in range(
                    step_size, int(num_samples * MAX_CACHE_SIZE), step_size
                ):
                    args.append(
                        (
                            dataset_name,
                            cache_name,
                            index_name,
                            same_embed_distance,
                            num_samples,
                            cache_size,
                            batch_size,
                            count_nn,
                        )
                    )

        results = {}
        Pool = ProcessPoolExecutor if NUM_PROCS > 1 else ThreadPoolExecutor
        with Pool(NUM_PROCS) as executor:
            futures = [executor.submit(run, arg) for arg in args]
            for future in as_completed(futures):
                result = future.result()
                for prop_name, prop in result.items():
                    _cache_name = result["Cache Name"]
                    cache_index = result["Index"]
                    cache_name = f"{_cache_name}_{cache_index}"
                    cache_size = result["Cache Size"]
                    if prop_name == "Cache Name":
                        continue
                    if prop_name not in results:
                        results[prop_name] = {}
                    if cache_name not in results[prop_name]:
                        results[prop_name][cache_name] = {}
                    results[prop_name][cache_name][cache_size] = prop
        plot(dataset_name, results)

def compare_crvb_and_nrvb():
    if os.path.exists("results.json"):
        os.remove("results.json")
    
    batch_size = 1
    count_nn = 1
    num_samples = 1000
    CAHCE_SIZE = 0.1
    dim = 384
    MIN_DIS = 0.5
    MAX_DIS = 1.0
    COUNT_DIS = 11

    faiss_indices_names = {"flat-naive"}
    caches_names = {"NaiveRVB", "ClusterRVB"}

    processor = Processor(NUM_PROCS)
    
    args = []
    for dataset_name, _ in get_embeds_paths():
        for same_embed_distance in np.linspace(MIN_DIS, MAX_DIS, COUNT_DIS):
            for cache_name in caches_names:
                for index_name in faiss_indices_names:
                    cache_size = int(num_samples * CAHCE_SIZE)
                    processor.submit(
                        dataset_name,
                            cache_name,
                            index_name,
                            same_embed_distance,
                            num_samples,
                            cache_size,
                            batch_size,
                            count_nn,
                            "results.json"
                    )
    processor.run()
    
def plot_compare_crvb_and_nrvb():
    df = []
    with open("results.json", "r") as f:
        for line in f:
            line = line.strip()
            if line:
                df.append(json.loads(line))
    df = pd.DataFrame(df)

    df_data = []  # was {} — needs to be a list
    for dataset_name in set(df['Dataset']):
        for eps in set(df['Same Embed Distance']):
            nrvb_rows = df[(df["Dataset"] == dataset_name) & (df["Same Embed Distance"] == eps) & (df["Cache Name"] == "NaiveRVB")]
            crvb_rows = df[(df["Dataset"] == dataset_name) & (df["Same Embed Distance"] == eps) & (df["Cache Name"] == "ClusterRVB")]
            if nrvb_rows.empty or crvb_rows.empty:
                continue
            hr_nrvb = nrvb_rows['AtLeast1@K'].iloc[0]
            hr_crvb = crvb_rows['AtLeast1@K'].iloc[0]
            ratio = (hr_nrvb / hr_crvb) if hr_crvb else float('nan')
            df_data.append({"dataset": dataset_name, "eps": eps, "ratio": ratio})

    heat = pd.DataFrame(df_data).pivot(index="dataset", columns="eps", values="ratio")
    plt.figure()
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="YlOrRd", cbar=True)

    plt.title("NaiveRVB / ClusterRVB (AtLeast1@K)")
    plt.show()

if __name__ == "__main__":
    compare_crvb_and_nrvb()
    plot_compare_crvb_and_nrvb()
