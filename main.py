import json
from random import sample
import time
from typing import List
from matplotlib import pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from vector_store import VectorStore

from dimensionality_reduction import *

def embed_strings(strings: List[str], model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(strings, convert_to_numpy=True)
    return embeddings.astype(np.float64)

def load_data_long(n: None):
    with open("prompts.json", "r") as f:
        data = json.load(f)
        origin = [v["text_a"] for v in data if v["label"] == 1]
        similar = [v["text_b"] for v in data if v["label"] == 1]
        if n:
            indices = sample(list(range(len(origin))), n)
            origin = [origin[i] for i in indices]
            similar = [similar[i] for i in indices]
        return origin, similar

def load_data_short():
    with open("mock_data.json", "r") as f:
        return json.load(f)

def plot(results):
    """
    Plots the results of dimensionality reduction experiments.

    Args:
        results (dict): A dictionary where keys are reduction technique names and values are
                        lists of tuples (dim_reduce, score).
    """
    
    for prop_name, prop_results in results.items():
        plt.figure(figsize=(10, 6))

        for reduce_name, data in prop_results.items():
            dims, scores = zip(*data)  # Unzip dimensions and scores
            plt.plot(dims, scores, label=reduce_name, marker="o")

        plt.xlabel("Reduced Dimensionality")
        plt.ylabel(prop_name)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{prop_name}.png")

from scipy.spatial.distance import cdist

def normalize_matrix(matrix):
    scaler = StandardScaler()
    flat_matrix = matrix.flatten().reshape(-1, 1)
    normalized_flat = scaler.fit_transform(flat_matrix).reshape(matrix.shape)
    return normalized_flat

def compare_distances(original, reduced):
    distances_original = normalize_matrix(cdist(original, original, metric='euclidean'))
    distances_reduced = normalize_matrix(cdist(reduced, reduced, metric='euclidean'))
    max_element = 2 * max(distances_original.max(), distances_reduced.max())
    np.fill_diagonal(distances_original, max_element)
    np.fill_diagonal(distances_reduced, max_element)
    closest_original = np.argmin(distances_original, axis=1)
    closest_reduced = np.argmin(distances_reduced, axis=1)
    accuracy_closest = np.sum(closest_original == closest_reduced) / len(closest_original)
    difference = distances_original - distances_reduced
    distance_mse = np.mean(difference ** 2)
    return accuracy_closest, distance_mse

def main():
    strings_origin, strings_similar = load_data_long(2000)
    embeds = embed_strings(strings_origin + strings_similar)
    embeds_origin = embeds[:len(strings_origin)]
    embeds_similar = embeds[len(strings_origin):]
    dim = embeds_origin[0].shape[0]
    
    reduces = {
        "PCA": reduce_pca,
        # "NORMALIZE": autoscale,
        # "TSNE": reduce_tsne,
        # "SVD": reduce_svd,
        # "MSD": reduce_msd,
        # "MSD_NOSCALE": reduce_msd_no_scale
        # "FactorAnalysis": reduce_fa,
        # "ICA": reduce_ica,
        # "GaussRandProj": reduce_grp,
        # "NOOP": reduce_noop
    }
    
    step_size = 10
    results = {}
    
    for reduce_name, reduce in reduces.items():
        for dim_reduce in list(range(1, dim, step_size)) + [dim]:
            store = VectorStore(dim_reduce)
            t0 = time.time()
            reducer, embeds_origin_reduce = reduce(embeds_origin, dim, dim_reduce)
            embeds_similar_reduce = reducer.apply(embeds_similar)
            t1 = time.time()
            store.add(embeds_origin_reduce, strings_origin)
            t2 = time.time()
            strings_pred, embeds_pred, distances, indices = store.search(embeds_similar_reduce, 1)
            t3 = time.time()
            strings_actual = strings_origin
            compare_actuals_preds = np.all(embeds_pred == embeds_origin_reduce, axis=1)
            nn_accuracy_vectorstore = np.sum(compare_actuals_preds) / len(compare_actuals_preds)
            embeds_reduce = np.concat((embeds_origin_reduce, embeds_similar_reduce), axis=0)
            nn_accuracy_distance, mse_distance = compare_distances(embeds, embeds_reduce)
            iter_results = {
                "time_total": t3-t0,
                "time_reduce": t1-t0,
                "time_add": t2-t1,
                "time_search": t3-t2,
                "nn_accuracy_vectorstore": nn_accuracy_vectorstore,
                "nn_accuracy_distance": nn_accuracy_distance,
                "mse_distance": mse_distance
            }
            print(reduce_name, dim_reduce, iter_results)
            for iter_result_name, iter_result in iter_results.items():
                if iter_result_name not in results:
                    results[iter_result_name] = {}
                if reduce_name not in results[iter_result_name]:
                    results[iter_result_name][reduce_name] = []
                results[iter_result_name][reduce_name].append((dim_reduce, iter_result))
    plot(results)

if __name__=="__main__":
    main()