from sklearn.decomposition import PCA
import itertools
import faiss
import numpy as np
from collections import deque

def get_N_or_less_from_generator(stream, N):
    return list(itertools.islice(stream, N))

def reduce_dim(vectors, dim_output = 32, fit_size=100000):
    dim_input = vectors.shape[1]
    if dim_output >= dim_input:
        return vectors
    vectors_train = vectors[:fit_size]
    pca = PCA(n_components=dim_output)
    pca.fit(vectors_train)
    vectors_transformed = pca.transform(vectors)
    return vectors_transformed

def greedy_cluster_faiss(embeds: np.ndarray, d: float) -> np.ndarray:
    n, dim = embeds.shape
    assert embeds.dtype == np.float32

    radius = d ** 2
    cluster_ids = np.full(n, -1, dtype=np.int32)
    cluster_centers = []
    cluster_index = faiss.IndexFlatL2(dim)

    next_cluster_id = 0

    for i in range(n):
        x = embeds[i:i+1]

        if cluster_centers:
            # Search existing cluster centers
            D, I = cluster_index.search(x, 1)
            if D[0][0] <= radius:
                cluster_ids[i] = I[0][0]  # assign to nearest cluster center
                continue

        # No close cluster found → new cluster
        cluster_ids[i] = next_cluster_id
        cluster_index.add(x)
        cluster_centers.append(i)
        next_cluster_id += 1

    return cluster_ids

def cluster_complete_linkage_faiss(embeds: np.ndarray, d: float) -> np.ndarray:
    n, dim = embeds.shape
    assert embeds.dtype == np.float32

    index = faiss.IndexFlatL2(dim)
    index.add(embeds)
    lims, _, indices = index.range_search(embeds, d)

    neighbors = [set(indices[lims[i]:lims[i+1]]) - {i} for i in range(n)]

    sorted_indices = sorted(range(n), key=lambda i: -len(neighbors[i]))

    cluster_count = 0
    cluster_ids = np.full(n, -1, dtype=int)
    assigned = set()

    for i in sorted_indices:
        if i in assigned:
            continue
        cluster = {i}
        candidates = neighbors[i] - assigned

        for c in list(candidates):
            if all(c in neighbors[member] for member in cluster): # only need each vector to be close enough to all vectors that come after it
                cluster.add(c)

        for member in cluster:
            cluster_ids[member] = cluster_count
        assigned.update(cluster)
        cluster_count += 1

    return cluster_ids

def cluster_embeddings_faiss(embeds: np.ndarray, d: float) -> np.ndarray:
    n, dim = embeds.shape
    assert embeds.dtype == np.float32

    # Step 1: Create FAISS index for fast range search
    index = faiss.IndexFlatL2(dim)
    index.add(embeds)

    # Step 2: Find all neighbors within distance d (squared)
    # Returns: lims, distances, indices
    radius = d ** 2
    lims, D, I = index.range_search(embeds, radius)

    # Step 3: Build adjacency list (undirected)
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        for j in range(lims[i], lims[i+1]):
            k = I[j]
            if i != k:  # skip self-loops
                neighbors[i].append(k)
                neighbors[k].append(i)

    # Step 4: Find connected components
    visited = np.full(n, False)
    cluster_ids = np.full(n, -1, dtype=int)
    current_cluster = 0

    for i in range(n):
        if visited[i]:
            continue
        # BFS or DFS from node i
        queue = deque([i])
        visited[i] = True
        cluster_ids[i] = current_cluster

        while queue:
            node = queue.popleft()
            for neighbor in neighbors[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    cluster_ids[neighbor] = current_cluster
                    queue.append(neighbor)

        current_cluster += 1

    return cluster_ids