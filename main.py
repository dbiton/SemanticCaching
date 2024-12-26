import json
from typing import List
from matplotlib import pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from vector_store import VectorStore

from dimensionality_reduction import *

def embed_strings(strings: List[str], model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(strings, convert_to_numpy=True)
    return embeddings

def load_data():
    with open("mock_data.json", "r") as f:
        return json.load(f)

def plot(results):
    """
    Plots the results of dimensionality reduction experiments.

    Args:
        results (dict): A dictionary where keys are reduction technique names and values are
                        lists of tuples (dim_reduce, score).
    """
    plt.figure(figsize=(10, 6))

    for reduce_name, data in results.items():
        dims, scores = zip(*data)  # Unzip dimensions and scores
        plt.plot(dims, scores, label=reduce_name, marker="o")

    plt.title("Dimensionality Reduction Performance")
    plt.xlabel("Reduced Dimensionality")
    plt.ylabel("Accuracy Score")
    plt.legend()
    plt.grid(True)
    plt.show()

   

def main():
    data = load_data()
    strings_origin = [v['origin'] for v in data]
    strings_similar = [v['similar'] for v in data]
    embeds_origin = embed_strings(strings_origin)
    embeds_similar = embed_strings(strings_similar)
    dim = embeds_origin[0].shape[0]
    
    reduces = {
        "PCA": reduce_pca,
        # "TSNE": reduce_tsne,
        "SVD": reduce_svd,
        "MSD": reduce_msd,
        "MSD_NOSCALE": reduce_msd_no_scale
        # "FactorAnalysis": reduce_fa,
        # "ICA": reduce_ica,
        # "GaussRandProj": reduce_grp,
        # "NOOP": reduce_noop
    }
    
    step_size = 10
    results = {reduce_name: [] for reduce_name in reduces}
    
    for reduce_name, reduce in reduces.items():
        for dim_reduce in range(dim, 1, -step_size):
            store = VectorStore(dim_reduce)
            embeds_origin_reduce = reduce(embeds_origin, dim_reduce)
            embeds_similar_reduce = reduce(embeds_similar, dim_reduce)
            store.add(embeds_origin_reduce, strings_origin)
            strings_pred, embeds_pred, distances = store.search(embeds_similar_reduce, 1)
            strings_actual = strings_origin
            compare_actuals_preds = np.all(embeds_pred == embeds_origin_reduce, axis=1)
            score = np.sum(compare_actuals_preds) / len(compare_actuals_preds)
            print(reduce_name, dim_reduce, score)
            results[reduce_name].append((dim_reduce, score))
    plot(results)

if __name__=="__main__":
    main()