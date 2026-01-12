import json
import math
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
    "MsMarco": "datasets/embeds_msmarco.pkl",
    "WildChat": "datasets/embeds_wildchat.pkl",
    "ELI5": "datasets/embeds_eli5.pkl",
    "NaturalQuestions": "datasets/embeds_nq.pkl",
    "StackOverflow": "datasets/embeds_stackoverflow.pkl",
    "Quora": "datasets/embeds_quora_qp.pkl",
    "MMLU": "datasets/embeds_mmlu.pkl",
    "TriviaQA": "datasets/embeds_triviaqa.pkl",
    "HotPotQA": "datasets/embeds_hotpotqa.pkl",
}

NUM_PROCS = 8


def plot():
    df = []
    with open("results.json", "r") as f:
        for line in f:
            line = line.strip()
            if line:
                df.append(json.loads(line))
    df = pd.DataFrame(df)

    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)

    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "v", "D", "<", ">", "X", "+"]

    ignore_policies = [
        "RR",
        "Surprisal",
        "LIFO",
        "FIFO"
    ]
    
    # --- Generate normal plots ---
    for dataset_name in set(df['Dataset']):
        df_dataset = df[(df["Dataset"] == dataset_name)]
        for prop_name in df.columns:
            plt.figure()
            i = 0
            handles, labels = [], []
            for cache_name in set(df_dataset["Cache Name"]):
                if cache_name in ignore_policies:
                    continue
                df_cache = df_dataset[df_dataset["Cache Name"] == cache_name].sort_values("Cache Size")
                cache_size = df_cache["Cache Size"].to_list()
                prop_values = df_cache[prop_name].to_list()
                h, = plt.plot(
                    cache_size,
                    prop_values,
                    label=cache_name,
                    marker=markers[i % len(markers)],
                    linestyle=linestyles[i % len(linestyles)],
                )
                handles.append(h)
                labels.append(cache_name)
                i += 1
            plt.xlabel("Cache Size")
            plt.ylabel(prop_name)
            plt.grid(True)
            plt.tight_layout()
            plt.legend()
            plt.savefig(os.path.join(figures_dir, f"{prop_name}_{dataset_name}.png"))
            plt.close()

    # --- Create standalone legend figure ---
    fig, ax = plt.subplots(figsize=(6, 0.5))  # adjust width/height
    ax.axis("off")
    # Reuse last handles/labels or regenerate from df if needed
    fig.legend(
        handles, labels,
        loc="center",
        ncol=min(len(labels), 7),  # wrap into multiple columns if many
        frameon=False
    )
    fig.savefig(os.path.join(figures_dir, "legend.png"), bbox_inches="tight")
    plt.close(fig)
    


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
    num_samples = 50000
    MAX_CACHE_SIZE = 0.1
    COUNT_STEPS = 10
    dim = 384
    same_embed_distance = 0.85

    faiss_indices_names = {"HotSwap"}
    caches_names = {
        "NaiveRVB",
        #"ClusterRVB",
        #"CoverOPT",
        "SphereLFU",
        #"SurprisalLFU",
        #"Surprisal",
        #"SampleCache",
        "LFU",
        #"MissLFU",
        "LRU",
        #"LRUK",
        #"DALFU",
        #"ARC",
        #"ClusterLRU",
        #"ClusterLFU",
        #"FIFO",
        #"LIFO",
        #"RR",
        #"DistanceLFU",
        #"RAP",
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
    num_samples = 5000
    CAHCE_SIZE = 0.1
    MIN_DIS = 0.5
    MAX_DIS = 1.0
    COUNT_DIS = 10
    
    faiss_indices_names = {"HotSwap"}
    caches_names = {"CoverOPT", "NaiveRVB", "ClusterRVB"}

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
    num_samples = 100000
    SAME_EMBED_DISTANCE = 0.5

    faiss_indices_names = {"milvus-standalone-hnsw", "milvus-standalone-ivf", "hnswlib", "HotSwap", "faiss", "milvus-standalone-flat"}
    caches_names = {"LFU"}

    processor = Processor(NUM_PROCS)

    # BATCH SIZE
    for dataset_name, _ in get_embeds_paths():
        for batch_size in np.linspace(MIN_BATCH_SIZE, MAX_BATCH_SIZE, COUNT_BATCH_SIZE):
            for cache_name in caches_names:
                for index_name in faiss_indices_names:
                    cache_size = int(num_samples * MAX_CACHE_RATIO)
                    curr_num_samples = cache_size * 5
                    processor.submit(
                        dataset_name,
                        cache_name,
                        index_name,
                        SAME_EMBED_DISTANCE,
                        curr_num_samples,
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
                    curr_num_samples = cache_size * 5
                    processor.submit(
                        dataset_name,
                        cache_name,
                        index_name,
                        SAME_EMBED_DISTANCE,
                        curr_num_samples,
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
                    curr_num_samples = cache_size * 5
                    processor.submit(
                        dataset_name,
                        cache_name,
                        index_name,
                        SAME_EMBED_DISTANCE,
                        curr_num_samples,
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
        for i, index_name in enumerate(["faiss", "HotSwap", "hnswlib", "milvus-standalone-flat", "milvus-standalone-hnsw", "milvus-standalone-ivf"]):
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
    # 1. Load Data
    data = []
    with open("results-crvb.json", "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # Convert directly to DataFrame
    df = pd.DataFrame(data)

    # 2. Filter for the three specific policies
    target_policies = ["NaiveRVB", "ClusterRVB", "CoverOPT"]
    df = df[df["Cache Name"].isin(target_policies)]
    
    # Ensure Hit Rate is numeric
    df["Hit Rate"] = pd.to_numeric(df["Hit Rate"])

    # 3. Setup "Small Multiples" Grid (One plot per dataset)
    datasets = sorted(df["Dataset"].unique())
    n_datasets = len(datasets)
    cols = 3
    rows = math.ceil(n_datasets / cols)
    
    # Create the figure
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True)
    axes = axes.flatten()

    # Define consistent styles
    styles = {
        "CoverOPT": {"color": "#1f77b4", "marker": "o"},
        "ClusterRVB": {"color": "#ff7f0e", "marker": "s"}, # Likely your TGRVB
        "NaiveRVB": {"color": "#2ca02c", "marker": "^"}   # Likely your VGRVB
    }

    # 4. Loop through datasets and plot Absolute Hit Rate
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        subset = df[df["Dataset"] == dataset]
        
        # Plot lines
        sns.lineplot(
            data=subset,
            x="Same Embed Distance",
            y="Hit Rate",
            hue="Cache Name",
            style="Cache Name",
            markers=True,
            dashes=False,
            palette={k: v["color"] for k, v in styles.items()},
            ax=ax
        )
        
        # Formatting
        ax.set_yscale("log")
        ax.set_title(dataset, fontweight='bold')
        ax.set_ylabel("Absolute Hit Rate")
        ax.set_xlabel(r"$D_{thresh}$")
        ax.set_ylim(0, 1.05)  # Standardize Y-axis to 0-100%
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.get_legend().remove()  # Hide individual legends to reduce clutter

    # 5. Hide unused subplots (if any)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # 6. Add a single Global Legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08) # Make space for the legend
    
    # Save and Show
    plt.savefig("figures/absolute_hit_rate_comparison.pdf", bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    #mv = MilvusVectorStore()
    #mv.drop_all()
    #recall()
    #main()
    #plot()
    
    #compare_vector_stores()
    #plot_compare_vector_stores()
    
    #compare_index_runtime()
    #plot_compare_index_runtime()
    compare_crvb()
    plot_compare_crvb()
