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
    num_samples = 100
    MAX_CACHE_SIZE = 0.1
    COUNT_STEPS = 10
    dim = 384
    same_embed_distance = 0.75

    faiss_indices_names = {"naive"}
    caches_names = {
        "NaiveRVB",
        "ClusterRVB",
        "SurprisalLFU",
        "Surprisal",
        "LFU",
        "LRU",
        "LRUK",
        "DALFU",
        "ARC",
        "ClusterLRU",
        "ClusterLFU",
        "DistanceLFU",
        "RAP",
        "SphereLFU",
    }

    processor = Processor(NUM_PROCS)
    for dataset_name, _ in get_embeds_paths():
        for cache_name in caches_names:
            step_size = int(num_samples * MAX_CACHE_SIZE // COUNT_STEPS)
            for index_name in faiss_indices_names:
                for cache_size in range(
                    step_size, int(num_samples * MAX_CACHE_SIZE), step_size
                ):
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


def compare_crvb_and_nrvb():
    batch_size = 1
    count_nn = 1
    num_samples = 16384
    CAHCE_SIZE = 0.1
    MIN_DIS = 0.5
    MAX_DIS = 1.0
    COUNT_DIS = 7

    faiss_indices_names = {"naive"}
    caches_names = {"NaiveRVB", "ClusterRVB"}

    processor = Processor(NUM_PROCS)

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
                        "results.json",
                    )
    processor.run()


def compare_batch_size():
    MIN_BATCH_SIZE = 1
    MAX_BATCH_SIZE = 16
    COUNT_BATCH_SIZE = 10
    count_nn = 1
    num_samples = 10000
    CACHE_RATIO = 0.1
    SAME_EMBED_DISTANCE = 0.75

    faiss_indices_names = {"naive", "faiss", "milvus-standalone"}  # more indices? HNSW?
    caches_names = {"LFU"}

    processor = Processor(NUM_PROCS)

    for dataset_name, _ in get_embeds_paths():
        for batch_size in np.linspace(MIN_BATCH_SIZE, MAX_BATCH_SIZE, COUNT_BATCH_SIZE):
            for cache_name in caches_names:
                for index_name in faiss_indices_names:
                    cache_size = int(num_samples * CACHE_RATIO)
                    processor.submit(
                        dataset_name,
                        cache_name,
                        index_name,
                        SAME_EMBED_DISTANCE,
                        num_samples,
                        cache_size,
                        int(batch_size),
                        count_nn,
                        "results_batch_size.json",
                    )
        break  # only one dataset
    processor.run()


def plot_compare_batch_size():
    df = []
    with open("results_batch_size.json", "r") as f:
        for line in f:
            line = line.strip()
            if line:
                df.append(json.loads(line))
    df = pd.DataFrame(df)
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "v", "D", "<", ">", "X", "+"]
    for i, index_name in enumerate(set(df["Index"])):
        df_index = df[(df["Index"] == index_name)]
        index_runtime = df_index.groupby("Batch Size", as_index=False)[
            "Throughput"
        ].mean()
        plt.plot(
            index_runtime["Batch Size"],
            index_runtime["Throughput"],
            label=index_name,
            marker=markers[i % len(linestyles)],
            linestyle=linestyles[i % len(linestyles)],
        )
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    figures_dir = "figures"
    plt.savefig(os.path.join(figures_dir, "batch-size.png"))


def compare_index_runtime():
    batch_size = 1
    count_nn = 1
    num_samples = 10000
    MIN_CACHE_RATIO = 0.01
    MAX_CACHE_RATIO = 0.1
    COUNT_RATIOS = 11
    SAME_EMBED_DISTANCE = 0.75

    faiss_indices_names = {"naive", "faiss", "milvus-standalone"}  # more indices? HNSW?
    caches_names = {"LFU"}

    processor = Processor(NUM_PROCS)

    for dataset_name, _ in get_embeds_paths():
        for cache_ratio in np.linspace(MIN_CACHE_RATIO, MAX_CACHE_RATIO, COUNT_RATIOS):
            for cache_name in caches_names:
                for index_name in faiss_indices_names:
                    cache_size = int(num_samples * cache_ratio)
                    processor.submit(
                        dataset_name,
                        cache_name,
                        index_name,
                        SAME_EMBED_DISTANCE,
                        num_samples,
                        cache_size,
                        batch_size,
                        count_nn,
                        "results_index_size.json",
                    )
        break  # only one dataset
    processor.run()


def plot_compare_index_runtime():
    df = []
    with open("results_index_size.json", "r") as f:
        for line in f:
            line = line.strip()
            if line:
                df.append(json.loads(line))
    df = pd.DataFrame(df)
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "v", "D", "<", ">", "X", "+"]
    for i, index_name in enumerate(set(df["Index"])):
        df_index = df[(df["Index"] == index_name)]
        index_runtime = df_index.groupby("Cache Size", as_index=False)[
            "Throughput"
        ].mean()
        plt.plot(
            index_runtime["Cache Size"],
            index_runtime["Throughput"],
            label=index_name,
            marker=markers[i % len(linestyles)],
            linestyle=linestyles[i % len(linestyles)],
        )
    plt.xlabel("Cache Size")
    plt.ylabel("Throughput")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    figures_dir = "figures"
    plt.savefig(os.path.join(figures_dir, "index-runtime.png"))


def plot_compare_crvb_and_nrvb():
    df = []
    with open("results.json", "r") as f:
        for line in f:
            line = line.strip()
            if line:
                df.append(json.loads(line))
    df = pd.DataFrame(df)

    df_data = []  # was {} — needs to be a list
    for dataset_name in set(df["Dataset"]):
        for eps in set(df["Same Embed Distance"]):
            nrvb_rows = df[
                (df["Dataset"] == dataset_name)
                & (df["Same Embed Distance"] == eps)
                & (df["Cache Name"] == "NaiveRVB")
            ]
            crvb_rows = df[
                (df["Dataset"] == dataset_name)
                & (df["Same Embed Distance"] == eps)
                & (df["Cache Name"] == "ClusterRVB")
            ]
            if nrvb_rows.empty or crvb_rows.empty:
                continue
            hr_nrvb = nrvb_rows["Hit Rate"].iloc[0]
            hr_crvb = crvb_rows["Hit Rate"].iloc[0]
            ratio = (hr_nrvb / hr_crvb) if hr_crvb else float("nan")
            df_data.append({"dataset": dataset_name, "eps": eps, "ratio": ratio})

    heat = pd.DataFrame(df_data).pivot(index="dataset", columns="eps", values="ratio")
    plt.figure()
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="YlOrRd", cbar=True)

    plt.title("NaiveRVB / ClusterRVB (AtLeast1@K)")
    plt.show()


if __name__ == "__main__":
    mv = MilvusVectorStore()
    mv.drop_all()
    main()
    # compare_batch_size()
    # plot_compare_batch_size()
    # compare_index_runtime()
    # plot_compare_index_runtime()
    # compare_crvb_and_nrvb()
    # plot_compare_crvb_and_nrvb()
