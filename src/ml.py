import os
import pickle
import numpy as np
import faiss
from sklearn.decomposition import PCA
import tqdm

from OPT import RLB_Reg, RelaxedLearnedOPT, RelaxedOPT
from src.util.reduce_dim import reduce_dim
from random import sample
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    cohen_kappa_score, balanced_accuracy_score,
    classification_report
)
from cache import LFU, LRU
import random

def binary_classification_stats(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    # Confusion‑matrix layout: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)          # a.k.a. sensitivity / TPR
    specificity = tn / (tn + fp) if (tn + fp) else 0                   # TNR
    f1        = f1_score(y_true, y_pred, zero_division=0)
    accuracy  = accuracy_score(y_true, y_pred)
    bal_acc   = balanced_accuracy_score(y_true, y_pred)
    mcc       = matthews_corrcoef(y_true, y_pred)
    kappa     = cohen_kappa_score(y_true, y_pred)

    return {
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall (sensitivity)": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "mcc": mcc,
        "cohen_kappa": kappa,
        "classification_report": classification_report(
            y_true, y_pred, target_names=["False", "True"], zero_division=0
        )
    }

def reduce_dimension(name, X):
    pca = PCA(n_components=16, svd_solver='randomized')
    X_reduced = pca.fit_transform(X)
    with open(f"{name}_reduced.pkl", "wb") as f:
        pickle.dump(X_reduced, f)

def save_embeds():
    SAME_EMBED_DISTANCE = .75
    for dataset_name, embeds in load_embeds():
        print(dataset_name)
        for size_exp in range(3, 10):
            size = 10 ** size_exp
            print(size)
            embeds_subset = embeds[:size]
            embeds_covers = create_embeds_covers(embeds_subset, SAME_EMBED_DISTANCE)
            file_name = f"{size}_{dataset_name}.json"
            with open(file_name, "wb") as f:
                pickle.dump(embeds_covers, f)
            if len(embeds) <= size:
                break

def load_embeds_covers():
    with open("1000000_StackOverflow.json", "rb") as f:
        return pickle.load(f)

dataset_filenames = {
    "OAsst": "datasets_text/embeds_oasst.pkl",
    #"persona": "datasets_text/embeds_persona.pkl",
    #"quora": "datasets_text/embeds_quora.pkl",
    #"WildChat": "datasets_text/embeds_chat.pkl",
    #"Bing": "datasets_text/embeds_bing.pkl",
    #"StackOverflow": "datasets_text/embeds_so.pkl",
    #"ComQA": "datasets_text/embeds_ComQA.pkl",
    #"Steam": "datasets/embeds_steam.pkl",
}

def load_embeds():
    for dataset_name, path in dataset_filenames.items():
        if not os.path.exists(path):
            print(f"Skipping \"{path}\" because it does not exist")
            continue
        with open(path, "rb") as f:
            embeds = pickle.load(f)
            yield dataset_name, embeds

def get_counts_hits(embeds_covers, curr_embed_id, embeds_ids):
    rel_covers = embeds_covers[embeds_ids]
    next_hits = np.full(len(embeds_ids), 0)
    for i_row, row in enumerate(rel_covers):
        idx = row.searchsorted(curr_embed_id, side='right')
        if len(row) > idx and len(row) > 0:
            next_hits[i_row] = len(row) - idx
    return next_hits

def get_next_hits(embeds_covers, curr_embed_id, embeds_ids):
    rel_covers = embeds_covers[embeds_ids]
    next_hits = np.full(len(embeds_ids), np.inf)
    for i_row, row in enumerate(rel_covers):
        idx = row.searchsorted(curr_embed_id, side='right')
        if len(row) > idx and len(row) > 0:
            next_hits[i_row] = row[idx]
    return next_hits

def get_prev_hits(embeds_covers, curr_embed_id, embeds_ids):
    rel_covers = embeds_covers[embeds_ids]
    prev_hits = []
    for row in rel_covers:
        idx = row.searchsorted(curr_embed_id, side='right')
        prev_hits.append(row[:idx])
    return np.array(prev_hits, dtype='object')

def create_embeds_covers(embeds, same_embed_distance):
    embeds = np.asarray(embeds, dtype=np.float32)
    n, d = embeds.shape
    cpu_index = faiss.IndexFlatL2(d)
    if faiss.get_num_gpus() > 0:
        gpu_res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)
    else:
        index = cpu_index
    index.add(embeds)
    threshold = same_embed_distance ** 2
    lims, _, indices = index.range_search(embeds, threshold)
    embeds_covers = [np.sort(indices[lims[i]:lims[i+1]][indices[lims[i]:lims[i+1]] > i]) for i in range(n)]
    return np.array(embeds_covers, dtype=object)

def yield_batches(lst, k):
    for i in range(0, len(lst), k):
        yield lst[i:i + k], list(range(i, min(i+k, len(lst))))


def test_regressor():
    DIM = 384
    DELTAS_COUNT = 4
    STREAM_SIZE = 100000
    CACHE_SIZE = 10000
    BELADY_BOUNDARY_COE = 2.0
    SAME_EMBED_DISTANCE = .5
    BATCH_SIZE = 10
    random.seed(42)
    for dataset_name, data in load_embeds():
        reg = RLB_Reg(CACHE_SIZE, DELTAS_COUNT, SAME_EMBED_DISTANCE, BELADY_BOUNDARY_COE, DIM)
        indices = sample(range(len(data['embeds'])), min(STREAM_SIZE, len(data['embeds'])))
        preps = [data['text'][i] for i in indices]
        embeds_actual = reduce_dim(np.array([data['embeds'][i] for i in indices]), DIM)
        embeds = list(zip(embeds_actual, preps))
        gdrs = []
        print(dataset_name)
        embeds_covers = create_embeds_covers(embeds_actual, SAME_EMBED_DISTANCE)
        evict_actual = []
        evict_predict = []
        maes = []
        for batch_embeds, i_embeds in yield_batches(embeds, BATCH_SIZE):
            print(i_embeds[-1], "/", STREAM_SIZE)
            embeds_texts = [v for (_, v) in batch_embeds]
            embeds_embeds = np.array([v for (v, _) in batch_embeds])
            if reg.train_counter > 0:
                belady_boundary = np.array(i_embeds) + reg.get_belady_boundary()
                actual = get_next_hits(embeds_covers, i_embeds[0], i_embeds)
                indices_replace = (actual - np.array(i_embeds)) > reg.get_belady_boundary()
                actual[indices_replace] = belady_boundary[indices_replace]
                actual_pred = np.log1p(actual - np.array(i_embeds))
                predict = reg.predict_tmp(i_embeds, embeds_embeds, embeds_texts, actual_pred)
                maes.append(np.abs(actual-predict))
                print(np.mean(maes))
                evict_actual += list(actual)
                evict_predict += list(predict)
                #gdrs.append(abs(actual-predict).mean())
                #print(np.mean(gdrs))
            reg.record_for_training(i_embeds, embeds_embeds, embeds_texts)
        x = 3


def show(y_true, y_pred):
    stats = binary_classification_stats(y_true, y_pred)

    # Pretty‑print a subset
    for k in ("TN", "FP", "FN", "TP", "accuracy", "precision", "recall (sensitivity)", "specificity", "f1"):
        print(f"{k:>16}: {stats[k]:.3f}" if isinstance(stats[k], float) else f"{k:>16}: {stats[k]}")

    print("\nFull classification report:\n", stats["classification_report"])
    
'''4300 / 5K cache'''
'''25000 / 20K cache'''


def test_policy():
    DIM = 384
    DELTAS_COUNT = 4
    STREAM_SIZE = 3000
    CACHE_SIZE = 1000
    BATCH_SIZE = 1
    COUNT_NN = 1
    SAME_EMBED_DISTANCE = 1.
    BELADY_BOUNDARY_COE = 1.0
    for dataset_name, data in load_embeds():
        indices = sample(range(len(data['embeds'])), min(STREAM_SIZE, len(data['embeds'])))
        preps = [data['text'][i] for i in indices]
        embeds_actual = reduce_dim(np.array([data['embeds'][i] for i in indices]), DIM)
        embeds = list(zip(embeds_actual, preps))
        embeds_covers = create_embeds_covers(embeds_actual, SAME_EMBED_DISTANCE)
        for cache in [
            LRU(SAME_EMBED_DISTANCE), 
            #RelaxedOPT(SAME_EMBED_DISTANCE, embeds_actual, BELADY_BOUNDARY_COE)]:
            RelaxedLearnedOPT(SAME_EMBED_DISTANCE, DELTAS_COUNT, 1, BELADY_BOUNDARY_COE, DIM)]:
            index = faiss.IndexIDMap2(faiss.IndexFlatL2(DIM))
            cache.initialize(CACHE_SIZE, index)
            count_evicts = 0
            count_good_evicts = 0
            for batch_embeds, i_embeds in yield_batches(embeds, BATCH_SIZE):
                next_i_embed = i_embeds[-1]
                embeds_texts = [v for (_, v) in batch_embeds]
                embeds_embeds = np.array([v for (v, _) in batch_embeds])
                iter_cache_hits, evicted_embeds_ids = cache.cache(embeds_embeds, i_embeds, COUNT_NN, embeds_texts)
                if len(evicted_embeds_ids) > 0:
                    count_evicts += len(evicted_embeds_ids)
                    next_hits = get_next_hits(embeds_covers, next_i_embed, evicted_embeds_ids) - i_embeds[-1]
                    count_good_evicts += len(np.where(next_hits >= BELADY_BOUNDARY_COE * CACHE_SIZE)[0])
                # pbar.update(len(batch_embeds))
            print(dataset_name, type(cache), "GDR:", count_good_evicts / count_evicts, count_good_evicts, count_evicts)
        
    
if __name__ == "__main__":
    test_policy()

                