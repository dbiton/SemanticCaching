from collections import Counter
import itertools
import pickle
import faiss
from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.decomposition import PCA
from scipy.stats import entropy
from .fetch_datasets import dataset_filenames, load_embeds
import seaborn as sns

from .reduce_dim import cluster_complete_linkage_faiss
from .density_estimator import DensityEstimator

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

def point_density(embeds, semantic_equiv_distance=0.75):
    """
    Efficient point-density using FAISS range search.
    Counts how many points lie within radius `semantic_equiv_distance`
    for each embedding.
    
    embeds: np.ndarray (n, d), float32 recommended
    returns: np.ndarray of shape (n,)
    """
    embeds = np.asarray(embeds).astype('float32')
    n, d = embeds.shape

    index = faiss.IndexFlatL2(d)  # exact L2 search
    index.add(embeds)

    # FAISS uses squared radius for L2
    radius_sq = semantic_equiv_distance ** 2

    # Range search: returns (lims, D, I)
    lims, D, I = index.range_search(embeds, radius_sq)

    # Convert FAISS prefix-sum style output to counts
    counts = np.empty(n, dtype=int)
    for i in range(n):
        counts[i] = lims[i+1] - lims[i]

    return counts

plt.rcParams.update({
    'figure.figsize': (3.0, 2.5),     # Matches the previous 3-per-row size
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'lines.markersize': 4,            # Smaller markers for the smaller plot
    'lines.linewidth': 1.0,           # Thinner lines for clarity
    'font.family': 'serif',           # LaTeX standard
    'axes.grid': True,
    'grid.alpha': 0.3
})

def plot_point_density_vs_rank(embeds, distribution_name):
    plt.figure()
    distances = np.arange(0.5, 1.5, 0.1)
    markers = itertools.cycle(["o", "s", "^", "v", "D", "<", ">", "X", "+", "*"])
    count_colors = len(distances)
    colors = sns.color_palette("tab10", count_colors)
    
    mark_indices = [int(v-1) for v in np.geomspace(1, len(embeds), 10)]
    
    for i, distance in enumerate(distances):
        counts = point_density(embeds, semantic_equiv_distance=distance)
        sorted_counts = sorted(counts, reverse=True)
        ranks = np.arange(1, len(sorted_counts) + 1)
        n_points = len(ranks)
        plt.plot(
            ranks, 
            sorted_counts, 
            alpha=0.7, 
            linewidth=2, 
            marker=next(markers),
            markevery=mark_indices,     # Plots a marker every 'step' points
            markersize=6,
            label=f'Distance = {distance:.1f}',
            color=colors[i]
        )
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Point Density (Count)', fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.savefig(f'{distribution_name}_point_density_distances.png', dpi=100)
    plt.close()
    fig_leg, ax_leg = plt.subplots(figsize=(10, 1)) 
    ax_leg.axis("off")
    fig_leg.legend(
        handles, labels,
        loc="center",
        ncol=5,  # Adjust columns as needed to fit width
        frameon=False,
        fontsize=10
    )
    fig_leg.savefig("density_legend.png", bbox_inches="tight", dpi=300)
    plt.close(fig_leg)


def cluster_sizes(embeds, distribution_name):
    clusters = cluster_complete_linkage_faiss(embeds, 1.2)
    clusters_sizes_dict = Counter(clusters)
    clusters_sizes = list(clusters_sizes_dict.values())
    print(f"Number of clusters: {len(clusters_sizes_dict)}")
    print(f"Mean cluster size: {np.mean(clusters_sizes)}")
    print(f"Std cluster size: {np.std(clusters_sizes)}")
    
    # Plot cluster size distribution
    sorted_sizes = sorted(clusters_sizes, reverse=True)
    ranks = np.arange(1, len(sorted_sizes) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(ranks, sorted_sizes, alpha=0.6)
    plt.xlabel('Rank')
    plt.ylabel('Cluster Size')
    plt.yscale('log')
    plt.xscale('log')
    plt.title(f'Cluster Size Distribution - {distribution_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.show()
    # This suggests a power law distribution, P(x) ~ x^-alpha
    # We can fit a linear model to log(rank) vs log(size)
    log_ranks = np.log(ranks)
    log_sizes = np.log(sorted_sizes)

    # Filter out any -inf values that might arise from log(0) if sorted_sizes contains 0
    valid_indices = np.isfinite(log_ranks) & np.isfinite(log_sizes)
    log_ranks = log_ranks[valid_indices]
    log_sizes = log_sizes[valid_indices]

    if len(log_ranks) > 1: # Need at least two points to fit a line
        # Perform linear regression
        # Using polyfit for simplicity, it returns coefficients [slope, intercept]
        coefficients = np.polyfit(log_ranks, log_sizes, 1)
        slope = coefficients[0]
        intercept = coefficients[1]

        # Plot the fitted line
        plt.plot(ranks, np.exp(intercept + slope * log_ranks), color='red', linestyle='--', label=f'Fit: y = {np.exp(intercept):.2f}x^{slope:.2f}')
        plt.legend()
    plt.savefig(f'{distribution_name}_loglog.jpg', dpi=100)
    plt.close()


def hypersphere_coverage(embeds, num_representatives, threshold):
    """
    Greedily select N hyperspheres that maximize coverage.
    Similar to max coverage problem - recalculates densest region after removing
    the first most dense area, and repeats num_representatives times.
    
    Args:
        embeds: np.ndarray of shape (n, d) - embeddings
        num_representatives: int - number of greedy selections to make
        threshold: float - radius of hyperspheres (semantic equivalence distance)
    
    Returns:
        covered_count: int - number of unique vectors covered by the hyperspheres
        coverage_ratio: float - ratio of covered vectors to total vectors
        representative_densities: np.ndarray - densities of the representative vectors
        representative_indices: np.ndarray - indices of the representative vectors
    """
    embeds = np.asarray(embeds).astype('float32')
    n, d = embeds.shape
    
    # Create FAISS index for range search
    index = faiss.IndexFlatL2(d)
    index.add(embeds)
    
    radius_sq = threshold ** 2
    uncovered_mask = np.ones(n, dtype=bool)  # Track which vectors are still uncovered
    covered_indices = set()
    representative_indices = []
    representative_densities = []
    
    for _ in range(num_representatives):
        # Calculate density only for uncovered vectors
        densities = point_density(embeds[uncovered_mask], semantic_equiv_distance=threshold)
        
        if len(densities) == 0:
            break  # No more uncovered vectors
        
        # Find the densest uncovered vector
        densest_uncovered_idx = np.argmax(densities)
        original_idx = np.where(uncovered_mask)[0][densest_uncovered_idx]
        
        representative_indices.append(original_idx)
        representative_densities.append(densities[densest_uncovered_idx])
        
        # Find all vectors covered by this hypersphere
        lims, D, I = index.range_search(embeds[original_idx:original_idx+1], radius_sq)
        newly_covered = set()
        for i in range(lims[0], lims[1]):
            newly_covered.add(I[i])
            covered_indices.add(I[i])
        
        # Mark these vectors as covered
        uncovered_mask[list(newly_covered)] = False
    
    covered_count = len(covered_indices)
    coverage_ratio = covered_count / n
    representative_densities = np.array(representative_densities)
    representative_indices = np.array(representative_indices)
    
    return covered_count, coverage_ratio, representative_densities, representative_indices


def compare_density_estimator(embeds, threshold=0.75):
    """
    Create, train, and compare a DensityEstimator to actual point density.
    
    Args:
        embeds: np.ndarray of shape (n, d) - embeddings to train and evaluate on
        cell_size: float - cell size parameter for DensityEstimator
        threshold: float - semantic equivalence distance for calculating actual density
    
    Returns:
        dict with metrics:
            - mae: mean absolute error
            - rmse: root mean squared error
            - pearson_correlation: Pearson correlation coefficient
            - r_squared: coefficient of determination
            - spearman_correlation: Spearman rank correlation
            - percent_correct_order: percentage of pairwise density orderings that match
            - actual_densities: ground truth densities
            - estimated_densities: estimator predictions
            - density_estimator: the trained DensityEstimator instance
    """
    from scipy.stats import spearmanr
    
    # Create and train the density estimator
    embeds_float32 = np.asarray(embeds, dtype=np.float32)
    dim = embeds_float32.shape[1]
    
    density_estimator = DensityEstimator(
        dim=384,
        width=100003, 
        reps=30,     
        hashes_per_rep=2, 
    )
    density_estimator.auto_tune(embeds_float32)
    density_estimator.update(embeds_float32)
    
    # Get density estimates from the trained estimator
    estimated_densities = density_estimator.query(embeds_float32)
    
    # Calculate actual point densities
    actual_densities = point_density(embeds, semantic_equiv_distance=threshold)
    
    # Ensure both arrays are the same shape and type
    estimated_densities = np.asarray(estimated_densities, dtype=np.float64)
    actual_densities = np.asarray(actual_densities, dtype=np.float64)
    
    if len(estimated_densities) != len(actual_densities):
        raise ValueError(f"Length mismatch: estimated ({len(estimated_densities)}) vs actual ({len(actual_densities)})")
    
    # Calculate metrics
    mae = np.mean(np.abs(estimated_densities - actual_densities))
    rmse = np.sqrt(np.mean((estimated_densities - actual_densities) ** 2))
    
    # Pearson correlation
    pearson_corr = np.corrcoef(estimated_densities, actual_densities)[0, 1]
    
    # R-squared (coefficient of determination)
    ss_res = np.sum((actual_densities - estimated_densities) ** 2)
    ss_tot = np.sum((actual_densities - np.mean(actual_densities)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Spearman rank correlation
    spearman_corr, _ = spearmanr(estimated_densities, actual_densities)
    
    # Percentage of correct pairwise orderings
    # Sample pairs to make this tractable for large datasets
    n = len(actual_densities)
    sample_size = min(10000, n * (n - 1) // 2)  # Limit to 10k pairs
    
    correct_orderings = 0
    for _ in range(sample_size):
        i, j = np.random.choice(n, 2, replace=False)
        actual_order = actual_densities[i] > actual_densities[j]
        estimated_order = estimated_densities[i] > estimated_densities[j]
        if actual_order == estimated_order:
            correct_orderings += 1
    
    percent_correct_order = (correct_orderings / sample_size) * 100
    
    results = {
        'mae': mae,
        'rmse': rmse,
        'pearson_correlation': pearson_corr,
        'r_squared': r_squared,
        'spearman_correlation': spearman_corr,
        'percent_correct_order': percent_correct_order,
        'actual_densities': actual_densities,
        'estimated_densities': estimated_densities,
        'density_estimator': density_estimator
    }
    
    print(results)
    
    return results

def hopkins_statistic(embeds, sample_fraction=1):
    """
    Calculates the Hopkins statistic to measure the clustering tendency of data.
    
    Formula: H = sum(W) / (sum(U) + sum(W))
    where:
      - W: Nearest neighbor distances from uniformly generated random points to real data
      - U: Nearest neighbor distances from sample real points to other real data
    
    Interpretation:
      - H near 0.5: Data is uniformly distributed (random)
      - H near 1.0: Data is highly clustered (values > 0.75 indicate clustering)
    """
    embeds = np.asarray(embeds).astype('float32')
    n, d = embeds.shape
    
    # 1. Determine sample size (m)
    m = int(n * sample_fraction)
    if m < 1: m = 1
    
    # 2. Build FAISS index for efficient search
    index = faiss.IndexFlatL2(d)
    index.add(embeds)
    
    # 3. Generate m synthetic points (Y) uniformly within the data's bounding box
    mins = np.min(embeds, axis=0)
    maxs = np.max(embeds, axis=0)
    
    # Generate random points in the hypercube defined by mins/maxs
    Y = np.random.uniform(mins, maxs, (m, d)).astype('float32')
    
    # 4. Calculate W: Distances from synthetic points (Y) to nearest real neighbor
    # search returns (distances_squared, indices)
    W_sq, _ = index.search(Y, 1)
    W = np.sqrt(W_sq).flatten() # Convert squared L2 to L2
    
    # 5. Calculate U: Distances from sampled real points (X) to their nearest neighbor
    # We sample indices from the real data
    real_indices = np.random.choice(n, m, replace=False)
    X = embeds[real_indices]
    
    # We search for 2 nearest neighbors because the 1st NN of a point X 
    # is X itself (distance 0). We need the 2nd one.
    U_sq, _ = index.search(X, 2)
    
    # Take the 2nd column (index 1) which is the nearest *other* neighbor
    U = np.sqrt(U_sq[:, 1]).flatten()
    
    # 6. Compute Hopkins Statistic
    sum_W = np.sum(W)
    sum_U = np.sum(U)
    
    # Avoid division by zero
    if sum_U + sum_W == 0:
        return 0.5
        
    H = sum_W / (sum_U + sum_W)
    
    print(f"Hopkins Statistic: {H:.4f}")
    return H


if __name__ == "__main__":

    results = {}

    # Calculate metrics and store them
    for dataset_name in dataset_filenames:
        embeds, _ = load_embeds(dataset_name, 50000)
        # hypersphere_coverage(embeds, 1000, 1.0)
        plot_point_density_vs_rank(embeds, dataset_name)
        hopkins = hopkins_statistic(embeds)
        
        # compare_density_estimator(embeds)
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
            "std_cluster_size": std_cluster_size,
            "hopkins": hopkins,
        }

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
    for metric_key in results[dataset_names[0]].keys():
        row_values = []
        for ds_name in dataset_names:
            # Format the number (4 decimal places)
            val = results[ds_name][metric_key]
            row_values.append(f"{val:.4f}")
        metric_name = metric_key.replace("_", " ").title()
        row_string = metric_name + " & " + " & ".join(row_values) + r" \\"
        print(row_string)

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Pairwise similarity, distance, and PCA entropy metrics (rows) across datasets (columns).}")
    print(r"\label{tab:embedding_stats_transposed}")
    print(r"\end{table}")
