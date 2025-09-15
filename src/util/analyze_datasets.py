from collections import Counter
import pickle
import faiss
from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.decomposition import PCA
from scipy.stats import entropy
import sys

def cluster_complete_linkage_faiss(embeds: np.ndarray, d: float) -> np.ndarray:
    n, dim = embeds.shape
    assert embeds.dtype == np.float32 and embeds.flags['C_CONTIGUOUS']
    r2 = float(d * d)  # FAISS uses squared L2

    index = faiss.IndexFlatL2(dim)
    index.add(embeds)
    lims, _, idx = index.range_search(embeds, r2)  # strict < r2 inside FAISS

    # Build neighbor sets INCLUDING self to simplify intersections
    neighbors = [set(idx[lims[i]:lims[i+1]]) | {i} for i in range(n)]
    deg = np.fromiter((len(s) - 1 for s in neighbors), count=n, dtype=np.int32)

    # Process low-degree vertices first (shrinks eligible sets faster)
    order = np.argsort(deg)  # ascending degree

    cluster_ids = -np.ones(n, dtype=np.int64)
    assigned: set[int] = set()
    cid = 0

    for i in order:
        if i in assigned:
            continue

        cluster = {i}
        # Start with all neighbors of i (including i), remove already assigned
        eligible = neighbors[i] - assigned

        # Repeatedly add a vertex that is compatible with everything chosen so far.
        # Compatibility is guaranteed by intersecting eligibility with its neighbors.
        while True:
            # don't reconsider current cluster members
            eligible -= cluster
            if not eligible:
                break
            # Heuristic: pick the smallest-degree vertex in the current eligible set
            c = min(eligible, key=deg.__getitem__)
            cluster.add(c)
            eligible &= neighbors[c]   # tighten to vertices adjacent to all in cluster

        for u in cluster:
            cluster_ids[u] = cid
        assigned.update(cluster)
        cid += 1

    return cluster_ids

def calculate_pairwise_statistics(embeddings):
    """
    Calculate the mean and standard deviation of pairwise cosine similarity 
    and Euclidean distance for the given embeddings.
    """
    # Normalize embeddings for cosine similarity calculation
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Pairwise cosine similarity
    cosine_sim = cosine_similarity(normalized_embeddings)
    cosine_sim_mean = np.mean(cosine_sim)
    cosine_sim_std = np.std(cosine_sim)
    
    # Pairwise Euclidean distance
    euclidean_dist = pairwise_distances(embeddings, metric='euclidean')
    euclidean_dist_mean = np.mean(euclidean_dist)
    euclidean_dist_std = np.std(euclidean_dist)
    
    return cosine_sim_mean, cosine_sim_std, euclidean_dist_mean, euclidean_dist_std

def plot_pca(embeds, fig_name, bins=100):
    """Perform PCA to 2D, then plot a 2D histogram of the point density."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(embeds)

    plt.figure(figsize=(8, 6))
    # Plot 2D histogram
    plt.hist2d(X_pca[:, 0], X_pca[:, 1], bins=bins, cmap='viridis')
    
    # Add a colorbar to show density scale
    plt.colorbar(label='Point Count')
    
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA with 2D Histogram')
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()


def plot_tsne(embeds, fig_name, bins=100, perplexity=30):
    """Perform t-SNE to 2D, then plot a 2D histogram of the point density."""
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(embeds)

    plt.figure(figsize=(8, 6))
    # Plot 2D histogram
    plt.hist2d(X_tsne[:, 0], X_tsne[:, 1], bins=bins, cmap='plasma')
    
    # Add a colorbar to show density scale
    plt.colorbar(label='Point Count')
    
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE with 2D Histogram')
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()

def calculate_entropy_of_principal_components(embeddings):
    """
    Calculate the entropy of the principal components of the given embeddings.
    """
    # Perform PCA
    pca = PCA()
    pca.fit(embeddings)
    explained_variance_ratio = pca.explained_variance_ratio_
    
    # Calculate entropy
    pca_entropy = entropy(explained_variance_ratio, base=2)  # Base 2 for bit entropy
    return pca_entropy

datasets_paths = {
    "MsMarco": "C:/Projects/DimCache/datasets/embeds_msmarco.pkl",
    "WildChat": "C:/Projects/DimCache/datasets/embeds_wildchat.pkl",
    "ELI5": "C:/Projects/DimCache/datasets/embeds_eli5.pkl",
    "NaturalQuestions": "C:/Projects/DimCache/datasets/embeds_nq.pkl",
    "StackOverflow": "C:/Projects/DimCache/datasets/embeds_stackoverflow.pkl",
    "Quora": "C:/Projects/DimCache/datasets/embeds_quora_qp.pkl",
}

results = {}

# Calculate metrics and store them
for dataset_name, file_path in datasets_paths.items():
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # Use a subset of embeddings if needed
    embeds = data['normalized_embeds'][:75000]
    #plot_pca(embeds, f"{dataset_name}_pca.png")
    #plot_tsne(embeds, f"{dataset_name}_tsne.png")
    clusters_ids = cluster_complete_linkage_faiss(embeds, 0.5)
    clusters_sizes_dict = Counter(clusters_ids)
    clusters_sizes = list(clusters_sizes_dict.values())
    mean_cluster_size = np.mean(clusters_sizes)
    std_cluster_size = np.std(clusters_sizes)
    
    # Calculate the pairwise statistics
    cs_mean, cs_std, ed_mean, ed_std = calculate_pairwise_statistics(embeds)
    
    # Calculate entropy of principal components
    pca_ent = calculate_entropy_of_principal_components(embeds)
    
    # Store results in a dictionary
    results[dataset_name] = {
        "cosine_sim_mean": cs_mean,
        "cosine_sim_std": cs_std,
        "euclidean_dist_mean": ed_mean,
        "euclidean_dist_std": ed_std,
        "pca_entropy": pca_ent,
        "mean_cluster_size": mean_cluster_size,
        "std_cluster_size": std_cluster_size
    }

# Define the order of metrics and the corresponding row labels
metrics_order = [
    ("cosine_sim_mean", "Mean Cosine Sim."),
    ("cosine_sim_std", "Std Cosine Sim."),
    ("euclidean_dist_mean", "Mean Eucl. Dist."),
    ("euclidean_dist_std", "Std Eucl. Dist."),
    ("pca_entropy", "PCA Entropy"),
    ("mean_cluster_size", "Mean Cluster Size"),
    ("std_cluster_size", "Std Cluster Size"),
]

# Get the list of datasets for columns
dataset_names = list(results.keys())

# Print results in a LaTeX table with metrics as rows and datasets as columns
print(r"\begin{table}[ht]")
print(r"\centering")

# Dynamically build the column header: one column for the metric labels + one per dataset
column_headers = " & " + " & ".join(dataset_names) + r" \\"
num_cols = len(dataset_names) + 1

print(r"\begin{tabular}{" + "l" + "c" * (num_cols - 1) + "}")
print(r"\hline")
print(column_headers)
print(r"\hline")

# Print each metric as a row
for metric_key, metric_label in metrics_order:
    row_values = []
    for ds_name in dataset_names:
        # Format the number (4 decimal places)
        val = results[ds_name][metric_key]
        row_values.append(f"{val:.4f}")
    
    row_string = metric_label + " & " + " & ".join(row_values) + r" \\"
    print(row_string)

print(r"\hline")
print(r"\end{tabular}")
print(r"\caption{Pairwise similarity, distance, and PCA entropy metrics (rows) across datasets (columns).}")
print(r"\label{tab:embedding_stats_transposed}")
print(r"\end{table}")
