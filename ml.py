import os
import pickle
import time
import numpy as np
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
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

def next_hits(embeds_covers, curr_embed_id, embeds_ids):
    """
    For each index in embeds_ids, find the index of the next 'hit' (where the cover is 1).
    If no hit exists in the future window, return np.inf.
    """
    start = curr_embed_id + 1
    n = embeds_covers.shape[1]
    # If there is no "future" (e.g. last embedding), return np.inf for all.
    if start >= n:
        return np.full(len(embeds_ids), np.inf)
    # Only consider columns from start onward.
    v = embeds_covers[embeds_ids, start:]
    # Create a boolean mask for rows that contain at least one hit.
    has_hit = np.any(v, axis=1)
    # Initialize result with np.inf
    hit_indices = np.full(len(embeds_ids), np.inf)
    if np.any(has_hit):
        # np.argmax returns the index of the first maximum along the axis.
        first_hits = np.argmax(v, axis=1)
        hit_indices[has_hit] = first_hits[has_hit]
    return hit_indices + start

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
    lims, distances, indices = index.range_search(embeds, threshold)
    embeds_covers = np.zeros((n, n), dtype=int)
    for i in range(n):
        start = lims[i]
        end = lims[i + 1]
        for pos in range(start, end):
            j = indices[pos]
            if j > i:
                embeds_covers[i, j] = 1
    return embeds_covers

if __name__ == "__main__":
    clf = XGBClassifier(objective='binary:logistic', random_state=42)
    same_embed_distance = 1.0
    for dataset_name, embeds in load_embeds():
        print(f"Processing dataset: {dataset_name}")
        embeds_covers = create_embeds_covers_faiss(embeds, same_embed_distance)
        step = 1000
        X_chunks = []
        y_chunks = []
        for i in range(0, len(embeds), step):
            embeds_chunk = embeds[i:i+step]
            X_chunks.append(embeds_chunk)
            upper = min(i + step, len(embeds))
            res = next_hits(embeds_covers, i, list(range(i, upper))) < upper
            y_chunks.append(res)
        print(len(X_chunks), "chunks created")
        X_flat = np.vstack(X_chunks)
        y_flat = np.concatenate(y_chunks)
        if X_flat.shape[0] != y_flat.shape[0]:
            raise ValueError("Mismatch between number of samples in X and y")
        print(f"Dataset {dataset_name}: {X_flat.shape[0]} samples with {X_flat.shape[1]} features")
        X_train, X_test, y_train, y_test = train_test_split(X_flat, y_flat, test_size=0.2, random_state=42)
        clf.fit(X_train, y_train)
        print(f"Finished training on dataset: {dataset_name}\n")
        y_pred = clf.predict(X_test)
        accuracy = precision_score(y_test, y_pred)
        print(f"Dataset {dataset_name} accuracy on validation set: {accuracy:.4f}\n")

