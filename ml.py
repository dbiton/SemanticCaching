from collections import defaultdict
import os
import pickle
import time
import numpy as np
from sklearn.metrics import mean_squared_error  
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from scipy.spatial import distance_matrix
import faiss

def load_embeds():
    filenames = {
        "Bing": "embeds_bing.pkl",
        "StackOverflow": "embeds_so.pkl",
        "WildChat": "embeds_chat.pkl"
    }
    embeds_dir = "datasets"
    for dataset_name, filename in filenames.items():
        path = os.path.join(embeds_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "rb") as f:
            embeds = pickle.load(f)
            embeds = np.array(embeds)
            # Validate that embeddings is a 2D array
            if embeds.ndim != 2:
                raise ValueError(
                    f"Embeddings for {dataset_name} must be a 2D array. Got shape: {embeds.shape}"
                )
            yield dataset_name, embeds

def get_next_hits(embeds_covers, curr_embed_id, embeds_ids):
    rel_covers = embeds_covers[embeds_ids]
    next_hits = np.full(len(embeds_ids), np.inf)
    for i_row, row in enumerate(rel_covers):
        idx = row.searchsorted(curr_embed_id, side='right')
        if idx < len(row):
            next_hits[i_row] = row[idx]
    return next_hits

def get_prev_hits(embeds_covers, curr_embed_id, embeds_ids):
    rel_covers = embeds_covers[embeds_ids]
    prev_hits = []
    for row in rel_covers:
        idx = row.searchsorted(curr_embed_id, side='right')
        prev_hits.append(row[:idx])
    return np.array(prev_hits, dtype='object')

def create_embeds_covers_distance_matrix(embeds, same_embed_distance):
    embeds_distances = distance_matrix(embeds, embeds)
    embeds_covers = (embeds_distances <= same_embed_distance).astype(int)
    tri_l = np.tril_indices_from(embeds_covers)
    embeds_covers[tri_l] = 0
    return embeds_covers

def create_embeds_covers_faiss(embeds, same_embed_distance):
    embeds = np.asarray(embeds, dtype=np.float32)
    n, d = embeds.shape
    index = faiss.IndexFlatL2(d)
    index.add(embeds)
    threshold = same_embed_distance ** 2
    lims, _, indices = index.range_search(embeds, threshold)
    embeds_covers = []
    for i in range(n):
        i_covers = np.sort(np.array([j for j in indices[lims[i]:lims[i+1]] if j > i]))
        embeds_covers.append(i_covers)
    return np.array(embeds_covers, dtype=object)

def extract_labels(embeds_covers, curr_embed_id, cached_embeds_ids, window_size):
    next_hits = get_next_hits(embeds_covers, curr_embed_id, cached_embeds_ids)
    next_hits = next_hits - curr_embed_id
    next_hits[next_hits > window_size] = 2 * window_size
    return np.log(next_hits)

def pad_array(arr, N, v):
    current_length = arr.shape[0]
    if current_length < N:
        pad_length = N - current_length
        return np.pad(arr, (pad_length, 0), mode='constant', constant_values=v)
    elif current_length > N:
        return arr[-N:]
    return arr

def embed_pca_to_cluster_id(self, cluster_diameter, embed_pca):
    unit_hypercube_diameter = np.sqrt(self.pca_dim)
    hypercube_side_length = cluster_diameter / unit_hypercube_diameter
    cluster_id = np.round(embed_pca / hypercube_side_length).astype(int)
    return tuple(cluster_id)

def extract_features(cached_embeds, curr_embed_id, cached_embeds_ids):
    DELTAS_COUNT = 8
    PCA_DIM = 9
    CLUSTER_DIAMETER = 1.0
    past_hits = get_prev_hits(embeds_covers, curr_embed_id, cached_embeds_ids)
    past_hits = [pad_array(v, DELTAS_COUNT, 1 + curr_embed_id) for v in past_hits]
    past_hits = np.array(past_hits)
    past_hits = past_hits - curr_embed_id
    deltas = np.array(past_hits)
    edc = np.stack([
        np.sum(2 ** (deltas / (2 ** (9 + i))), axis=1)
        for i in range(DELTAS_COUNT)
    ], axis=1)
    pca = faiss.PCAMatrix(cached_embeds, PCA_DIM)
    pca.train(embeds)
    assert pca.is_trained
    embeds_pca = pca.apply(embeds)
    clusters = defaultdict(int)
    for embed_id, embed_pca in zip(embeds_ids, embeds_pca):
        cluster_id = embed_pca_to_cluster_id(CLUSTER_DIAMETER, embed_pca)
        clusters[cluster_id] += 1
    static_features = None
    return np.hstack([deltas, edc])

if __name__ == "__main__":
    reg = XGBRegressor(objective='reg:squarederror', random_state=42)
    same_embed_distance = 0.5
    for dataset_name, embeds in load_embeds():
        embeds = embeds[:30000]
        print(f"Processing dataset: {dataset_name}")
        embeds_covers = create_embeds_covers_faiss(embeds, same_embed_distance)
        step = 1000
        X_chunks = []
        y_chunks = []
        for i in range(0, len(embeds) - 2*step, step):
            upper = min(i + step, len(embeds))
            windows_size = step
            embeds_ids = list(range(i, upper))
            embeds_chunk = embeds[i:i+step]
            features = extract_features(embeds_chunk, upper, embeds_ids)
            X_chunks.append(features)
            labels = extract_labels(embeds_covers, upper, embeds_ids, windows_size)
            y_chunks.append(labels)
            print("processed", i, "embeds")
        print(len(X_chunks), "chunks created")
        X_flat = np.vstack(X_chunks)
        y_flat = np.concatenate(y_chunks)
        if X_flat.shape[0] != y_flat.shape[0]:
            raise ValueError("Mismatch between number of samples in X and y")
        print(f"Dataset {dataset_name}: {X_flat.shape[0]} samples with {X_flat.shape[1]} features")
        X_train, X_test, y_train, y_test = train_test_split(X_flat, y_flat, test_size=0.2, random_state=42)
        reg.fit(X_train, y_train)
        print(f"Finished training on dataset: {dataset_name}\n")
        y_pred = reg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Dataset {dataset_name} MSE on validation set: {mse:.4f}\n")

