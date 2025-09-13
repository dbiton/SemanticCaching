import json
import sys
import pandas as pd

sys.path.append(".")

import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from vector_stores.milvus_interface import MilvusVectorStore
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

NUM_PROCS = 1


def plot():
    df = []
    with open("results.json", "r") as f:
        for line in f:
            line = line.strip()
            if line:
                df.append(json.loads(line))
    df = pd.DataFrame(df)
    for dataset_name in set(df['Dataset']):
        df_dataset = df[(df["Dataset"] == dataset_name)]
        for prop_name in df.columns:
            plt.figure()
            i = 0
            linestyles = ["-", "--", "-.", ":"]
            markers = ["o", "s", "^", "v", "D", "<", ">", "X", "+"]
            for cache_name in set(df_dataset["Cache Name"]):
                df_cache = df_dataset[(df_dataset["Cache Name"] == cache_name)]
                cache_size = list(df_cache['Cache Size'])
                prop_values = list(df_cache[prop_name])
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
            plt.yscale("log")
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


def recall():
    batch_size = 1
    count_nn = 10
    num_samples = 1000000
    MAX_CACHE_SIZE = 0.1
    COUNT_STEPS = 10
    dim = 384
    same_embed_distance = 0.75

    faiss_indices_names = {"HotSwap"}
    caches_names = {
        "NaiveRVB",
        "ClusterRVB",
        "SurprisalLFU",
        "Surprisal",
        "LFU",
        "MissLFU",
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
                    print(dataset_name)
                    processor.submit(
                        dataset_name,
                        cache_name,
                        index_name,
                        same_embed_distance,
                        num_samples,
                        cache_size,
                        batch_size,
                        count_nn,
                        "results-recall.json"
                    )
    processor.run()

def main():
    batch_size = 1
    count_nn = 1
    num_samples = 100000
    MAX_CACHE_SIZE = 0.1
    COUNT_STEPS = 10
    dim = 384
    same_embed_distance = 0.5

    faiss_indices_names = {"HotSwap"}
    caches_names = {
        "NaiveRVB",
        "ClusterRVB",
        "SurprisalLFU",
        "Surprisal",
        "LFU",
        "MissLFU",
        "LRU",
        "LRUK",
        "DALFU",
        "ARC",
        "ClusterLRU",
        "ClusterLFU",
        "FIFO",
        "LIFO",
        "RR",
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
                    print(dataset_name)
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


def compare_crvb():
    batch_size = 1
    count_nn = 1
    num_samples = 100000
    CAHCE_SIZE = 0.1
    MIN_DIS = 0.5
    MAX_DIS = 1.0
    COUNT_DIS = 7

    faiss_indices_names = {"HotSwap"}
    caches_names = {"NaiveRVB", "ClusterRVB", "LFU"}

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
                        "results-crvb.json",
                    )
    processor.run()


def compare_vector_stores():
    MIN_BATCH_SIZE = 1
    MAX_BATCH_SIZE = 16
    COUNT_BATCH_SIZE = 10
    MIN_NN_COUNT = 1
    MAX_NN_COUNT = 16
    COUNT_NN_COUNT = 10
    MIN_CACHE_RATIO = 0.01
    MAX_CACHE_RATIO = 0.1
    COUNT_CACHE_RATIO = 10
    num_samples = 10000
    SAME_EMBED_DISTANCE = 0.5

    faiss_indices_names = {"milvus-standalone-flat", "milvus-standalone-hnsw", "milvus-standalone-ivf", "HotSwap", "faiss", "hnswlib"}  # more indices? HNSW?
    caches_names = {"LFU"}

    processor = Processor(NUM_PROCS)

    # BATCH SIZE
    for dataset_name, _ in get_embeds_paths():
        for batch_size in np.linspace(MIN_BATCH_SIZE, MAX_BATCH_SIZE, COUNT_BATCH_SIZE):
            for cache_name in caches_names:
                for index_name in faiss_indices_names:
                    cache_size = int(num_samples * MAX_CACHE_RATIO)
                    processor.submit(
                        dataset_name,
                        cache_name,
                        index_name,
                        SAME_EMBED_DISTANCE,
                        num_samples,
                        cache_size,
                        int(batch_size),
                        MIN_NN_COUNT,
                        "results_vector_stores.json",
                    )
        break  # only one dataset
    # NN COUNT
    for dataset_name, _ in get_embeds_paths():
        for nn_count in np.linspace(MIN_NN_COUNT, MAX_NN_COUNT, COUNT_NN_COUNT):
            for cache_name in caches_names:
                for index_name in faiss_indices_names:
                    cache_size = int(num_samples * MAX_CACHE_RATIO)
                    processor.submit(
                        dataset_name,
                        cache_name,
                        index_name,
                        SAME_EMBED_DISTANCE,
                        num_samples,
                        cache_size,
                        MIN_BATCH_SIZE,
                        int(nn_count),
                        "results_vector_stores.json",
                    )
        break  # only one dataset
    # CACHE SIZE
    for dataset_name, _ in get_embeds_paths():
        for cache_ratio in np.linspace(MIN_CACHE_RATIO, MAX_CACHE_RATIO, COUNT_CACHE_RATIO):
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
                        MIN_BATCH_SIZE,
                        MIN_NN_COUNT,
                        "results_vector_stores.json",
                    )
        break  # only one dataset
    processor.run()


def plot_compare_vector_stores():
    df = []
    with open("results_vector_stores.json", "r") as f:
        for line in f:
            line = line.strip()
            if line:
                df.append(json.loads(line))
    df = pd.DataFrame(df)
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "v", "D", "<", ">", "X", "+"]
    
    # Batch Size
    for prop in ["Batch Size", "Count NN", "Cache Size", "Hit Rate"]:
        for i, index_name in enumerate(set(df["Index"])):
            df_index = df[(df["Index"] == index_name)]
            index_runtime = df_index.groupby(prop, as_index=False)[
                "Throughput"
            ].mean()
            plt.plot(
                index_runtime[prop],
                index_runtime["Throughput"],
                label=index_name,
                marker=markers[i % len(linestyles)],
                linestyle=linestyles[i % len(linestyles)],
            )
        plt.xlabel(prop)
        plt.ylabel("Throughput")
        plt.yscale("log")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        figures_dir = "figures"
        plt.savefig(os.path.join(figures_dir, f"{prop}.png"))
        plt.close()


def compare_index_runtime():
    batch_size = 1
    count_nn = 1
    num_samples = 1000000
    MIN_CACHE_RATIO = 0.01
    MAX_CACHE_RATIO = 0.1
    COUNT_RATIOS = 11
    SAME_EMBED_DISTANCE = 0.75

    faiss_indices_names = {"HotSwap", "faiss", "milvus-standalone"}  # more indices? HNSW?
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


def plot_compare_crvb():
    df = []
    with open("results-crvb.json", "r") as f:
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
            lfu_rows = df[
                (df["Dataset"] == dataset_name)
                & (df["Same Embed Distance"] == eps)
                & (df["Cache Name"] == "LFU")
            ]
            if nrvb_rows.empty or crvb_rows.empty:
                continue
            hr_nrvb = nrvb_rows["Hit Rate"].iloc[0]
            hr_crvb = crvb_rows["Hit Rate"].iloc[0]
            hr_lfu = lfu_rows["Hit Rate"].iloc[0]
            ratio_nrvb = (hr_nrvb / hr_crvb) if hr_crvb else float("nan")
            ratio_lfu = (hr_lfu / hr_crvb) if hr_crvb else float("nan")
            df_data.append({"Dataset": dataset_name, "D": eps, "ratio_nrvb": ratio_nrvb, "ratio_lfu": ratio_lfu})

    figures_dir = "figures"

    heat = pd.DataFrame(df_data).pivot(index="Dataset", columns="D", values="ratio_nrvb")
    plt.figure()
    ax = sns.heatmap(heat, annot=True, fmt=".2f", cmap="YlOrRd", cbar=True)
    plt.title("NRVB / CRVB Hit Rate Ratio")
    ax.set_xticklabels([f"{float(x.get_text()):.2f}" for x in ax.get_xticklabels()])
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"ratio_nrvb.png"))

    
    heat = pd.DataFrame(df_data).pivot(index="Dataset", columns="D", values="ratio_lfu")
    plt.figure()
    ax = sns.heatmap(heat, annot=True, fmt=".2f", cmap="YlOrRd", cbar=True)
    plt.title("LFU / CRVB Hit Rate Ratio")
    ax.set_xticklabels([f"{float(x.get_text()):.2f}" for x in ax.get_xticklabels()])
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"ratio_lfu.png"))

if __name__ == "__main__":
    #mv = MilvusVectorStore()
    #mv.drop_all()
    #recall()
    #main()
    #plot()
    
    compare_vector_stores()
    plot_compare_vector_stores()
    
    #compare_index_runtime()
    #plot_compare_index_runtime()
    # compare_crvb()
    # plot_compare_crvb()
