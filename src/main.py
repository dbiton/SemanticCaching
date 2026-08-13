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
from util.analyze_datasets import dataset_filenames

# use only text, embeds - remove normalized embeds


NUM_PROCS = 8


# --- 1. CONFIGURATION FOR OVERLEAF READABILITY ---
# A standard LaTeX column is ~6-7 inches wide. 
# For 3 plots per row, each image is roughly 2.0 - 2.3 inches wide.
# We generate them slightly larger (3.0) to allow for margin cropping without text becoming tiny.
plt.rcParams.update({
    'figure.figsize': (3.0, 2.5),     # Width, Height in inches
    'font.size': 8,                   # Base font size
    'axes.labelsize': 9,              # Axis labels (Cache Size, etc.)
    'axes.titlesize': 9,              # Title size
    'xtick.labelsize': 8,             # Tick numbers
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'lines.markersize': 4,            # Smaller markers for compact plots
    'lines.linewidth': 1.0,
    'font.family': 'serif',           # Matches standard LaTeX documents (Times/Computer Modern)
    'axes.grid': True,
    'grid.alpha': 0.3
})

def plot():
    df = []
    # Safety check for file existence
    if not os.path.exists("results_many.json"):
        print("results.json not found.")
        return

    with open("results_many.json", "r") as f:
        for line in f:
            line = line.strip()
            if line:
                df.append(json.loads(line))
    df = pd.DataFrame(df)

    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)

    markers = ["o", "s", "^", "v", "D", "<", ">", "X", "+", "*"]
    
    # Columns to EXCLUDE from plotting (metadata)
    ignore_cols = ["Dataset", "Cache Name", "Cache Size"] 
    ignore_policies = ["RR", "MissLFU", "LIFO"]

    # --- 2. CALCULATE GLOBAL LIMITS ---
    # To ensure axis starts at the same point, we need the global min/max
    # of the x-axis (Cache Size) across ALL datasets.
    global_min_x = df["Cache Size"].min()
    global_max_x = df["Cache Size"].max()
    
    # Optional: Add a small margin or snap to nearest log base if needed
    
    count_colors = len(set(df["Cache Name"]))
    # Using 'tab10' or 'Set1' is often more distinct than 'tab20' for lines
    colors = sns.color_palette("tab20", n_colors=max(count_colors, 1))

    # We need to collect handles/labels for the legend only once roughly
    legend_handles = []
    legend_labels = []

    distance_miss = 2
    df['Normalized Mean Hit Distance'] = df['Hit Rate'] * df['Mean Hit Distance'] + (1 - df['Hit Rate']) * distance_miss
    df['F1'] = df['Hit Rate'] * (2 - df['Mean Hit Distance']) / (2 + df['Hit Rate'] - df['Mean Hit Distance'])
    
    
    for threshold in set(df["Same Embed Distance"]):
        for dataset_name in set(df['Dataset']):
            df_dataset = df[(df["Dataset"] == dataset_name) & (df["Same Embed Distance"] == threshold)]
            
            # Iterate over columns, but skip metadata
            for prop_name in df.columns:
                if prop_name in ignore_cols:
                    continue

                plt.figure()
                
                # Reset handles for this specific plot (though we create a separate legend file later)
                i = 0
                sorted_caches = sorted(list(set(df_dataset["Cache Name"])))
                
                for cache_name in sorted_caches:
                    if cache_name in ignore_policies:
                        continue
                    
                    df_cache = df_dataset[df_dataset["Cache Name"] == cache_name].sort_values("Cache Size")
                    
                    cache_size = df_cache["Cache Size"].to_list()
                    prop_values = df_cache[prop_name].to_list()
                    
                    # Check if data exists
                    if not cache_size:
                        continue

                    h, = plt.plot(
                        cache_size,
                        prop_values,
                        label=cache_name,
                        marker=markers[i % len(markers)],
                        color=colors[i % len(colors)],
                    )
                    
                    # Capture for the global legend file (only need to do this once effectively)
                    if cache_name not in legend_labels:
                        legend_handles.append(h)
                        legend_labels.append(cache_name)
                    
                    i += 1

                # --- 3. AXIS SCALING & LABELING ---
                #plt.xlabel("Cache Size")
                #plt.ylabel(prop_name)
                
                plt.yscale("log")
                
                # Force the axis to start/end at the same points for all plots
                plt.xlim(global_min_x, global_max_x)

                plt.tight_layout(pad=0.5) # Reduces whitespace around the plot
                threshold_round = round(threshold, 2)
                plt.savefig(os.path.join(figures_dir, f"{prop_name}_{threshold_round}_{dataset_name}.png")) # PDF is better for LaTeX
                plt.close()

    # --- 4. GENERATE SEPARATE LEGEND ---
    # Sort legend to match the sorted cache names used in plotting
    if legend_handles:
        # Zip, sort by label, unzip
        hl_sorted = sorted(zip(legend_handles, legend_labels), key=lambda x: x[1])
        handles_sorted, labels_sorted = zip(*hl_sorted)

        fig_leg, ax_leg = plt.subplots(figsize=(6, 0.5))
        ax_leg.axis("off")
        fig_leg.legend(
            handles_sorted, 
            labels_sorted,
            loc="center",
            ncol=min(len(labels_sorted), 8),
            frameon=False
        )
        fig_leg.savefig(os.path.join(figures_dir, "legend.png"), bbox_inches="tight")
        plt.close(fig_leg)
    
def get_embeds_paths():
    return list(dataset_filenames.items())


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
        "RGRVB",
        "CRVB",
        "SurprisalLFU",
        "Surprisal",
        "LFU",
        "MissLFU",
        "LRU",
        "LRUK",
        "LFUDA",
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
    processor = Processor(NUM_PROCS)
    for same_embed_distance in np.arange(0.5, 1.0, 0.2):
        print(f"Running for same_embed_distance={same_embed_distance}")
        batch_size = 1
        count_nn = 1
        num_samples = 100000
        MAX_CACHE_SIZE = 0.1
        COUNT_STEPS = 8
        dim = 384

        faiss_indices_names = {"HotSwap"}
        caches_names = {
            "RGRVB",
            "CRVB",
            "FGRVB",
            "SphereLFU",
            "SurprisalLFU",
            "Surprisal",
            "LFU",
            #"MissLFU",
            "LRU",
            "LRUK",
            "LFUDA",
            "ARC",
            "ClusterLRU",
            "ClusterLFU",
            "FIFO",
            #"LIFO",
            #"RR",
            "DistanceLFU",
            "RAP",
        }

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
                            "results_many.json"
                        )
    processor.run()


def compare_crvb():
    batch_size = 1
    count_nn = 1
    num_samples = 100000
    CAHCE_SIZE = 0.1
    MIN_DIS = 0.5
    MAX_DIS = 1.0
    COUNT_DIS = 10
    
    faiss_indices_names = {"HotSwap"}
    caches_names = {"FGRVB", "RGRVB", "CRVB"}

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
    target_policies = ["RGRVB", "CRVB", "FGRVB"]
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
        "FGRVB": {"color": "#1f77b4", "marker": "o"},
        "CRVB": {"color": "#ff7f0e", "marker": "s"}, # Likely your TGRVB
        "RGRVB": {"color": "#2ca02c", "marker": "^"}   # Likely your VGRVB
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
    plot()
    
    #compare_vector_stores()
    #plot_compare_vector_stores()
    
    #compare_index_runtime()
    #plot_compare_index_runtime()
    #compare_crvb()
    #plot_compare_crvb()
