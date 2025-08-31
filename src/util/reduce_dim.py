import heapq
from typing import List, Set
from sklearn.decomposition import PCA
import itertools
import faiss
import numpy as np
from collections import deque
import networkx as nx
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


def cluster_by_coloring_dsat_faiss_heap(
    embeds: np.ndarray,
    d: float,
    strategy: str = "DSATUR"
) -> np.ndarray:
    n, dim = embeds.shape
    assert embeds.dtype == np.float32 and embeds.flags['C_CONTIGUOUS']

    # Build neighbor graph G with strict radius (< d^2)
    index = faiss.IndexFlatL2(dim)
    index.add(embeds)
    lims, D, I = index.range_search(embeds, float(d) * float(d))

    # Normalize FAISS outputs to numpy arrays
    lims = np.asarray(lims, dtype=np.int64)
    D    = np.asarray(D,    dtype=np.float32)
    I    = np.asarray(I,    dtype=np.int64)

    r2 = float(d) * float(d)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Add edges (i, j) when dist^2 < r2, j > i to avoid duplicates
    for i in range(n):
        start, end = int(lims[i]), int(lims[i + 1])
        for off in range(start, end):
            j = int(I[off])
            if j == i:
                continue
            if D[off] < r2 and j > i:
                G.add_edge(i, j)

    # Color the complement graph using DSATUR
    H = nx.complement(G)
    coloring = nx.algorithms.coloring.greedy_color(H, strategy=strategy)

    # Convert dict->array; normalize colors to 0..C-1 in order of first appearance
    assign = np.empty(n, dtype=int)
    remap = {}
    next_c = 0
    for v in range(n):
        c = int(coloring[v])
        if c not in remap:
            remap[c] = next_c
            next_c += 1
        assign[v] = remap[c]
    return assign