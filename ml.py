import os
import pickle
import numpy as np
import faiss
from sklearn.decomposition import PCA
import tqdm

from OPT import RLB_Reg, RelaxedLearnedOPT
from freq_reg import FreqReg
from reduce_dim import reduce_dim

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

def load_embeds():
    filenames = {
        "Steam": "embeds_steam.pkl",
        "Bing": "embeds_bing.pkl",
        "WildChat": "embeds_chat.pkl",
        "StackOverflow": "embeds_so.pkl",
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
    DIM = 100
    DELTAS_COUNT = 8
    STREAM_SIZE = 200000
    CACHE_SIZE = 64000
    BELADY_BOUNDARY_COE = 2.0
    SAME_EMBED_DISTANCE = .75
    BATCH_SIZE = 16
    for dataset_name, embeds in load_embeds():
        reg = RLB_Reg(CACHE_SIZE, DELTAS_COUNT, SAME_EMBED_DISTANCE, BELADY_BOUNDARY_COE, DIM)
        embeds = reduce_dim(embeds, DIM, STREAM_SIZE)
        gdrs = []
        print(dataset_name)
        embeds = embeds[:STREAM_SIZE]
        embeds_covers = create_embeds_covers(embeds, SAME_EMBED_DISTANCE)
        for batch_embeds, i_embeds in yield_batches(embeds, BATCH_SIZE):
            print(i_embeds[-1])
            if reg.is_trained():
                predict = reg.predict(i_embeds, batch_embeds)
                predict = 2**predict
                actual = get_next_hits(embeds_covers, i_embeds[0], i_embeds) - i_embeds
                actual[np.isposinf(actual)] = 2 ** reg.get_default_label()
                gdrs.append(abs(actual-predict).mean())
                print(np.mean(gdrs))
            reg.record_for_training(i_embeds, batch_embeds)
            

def test_policy():
    DIM = 384
    DELTAS_COUNT = 8
    STREAM_SIZE = 10000
    CACHE_SIZE = 640
    BATCH_SIZE = 20
    COUNT_NN = 1
    SAME_EMBED_DISTANCE = .75
    BELADY_BOUNDARY_COE = 2.0
    # cache = LFU(SAME_EMBED_DISTANCE)    
    cache = RelaxedLearnedOPT(SAME_EMBED_DISTANCE, DELTAS_COUNT, 1, BELADY_BOUNDARY_COE, DIM)    
    for dataset_name, embeds in load_embeds():
        embeds = embeds[:STREAM_SIZE]
        embeds = reduce_dim(embeds, DIM)
        pbar = tqdm.tqdm(total=len(embeds), desc=f"Processing {dataset_name}...")
        embeds_covers = create_embeds_covers(embeds, SAME_EMBED_DISTANCE)
        index = faiss.IndexIDMap2(faiss.IndexFlatL2(DIM))
        cache.initialize(CACHE_SIZE, index)
        count_evicts = 0
        count_good_evicts = 0
        for batch_embeds, i_embeds in yield_batches(embeds, BATCH_SIZE):
            next_i_embed = i_embeds[-1]
            iter_cache_hits, evicted_embeds_ids = cache.request(batch_embeds, i_embeds, COUNT_NN)
            if len(evicted_embeds_ids) > 0:
                count_evicts += len(evicted_embeds_ids)
                next_hits = get_next_hits(embeds_covers, next_i_embed, evicted_embeds_ids) - i_embeds[-1]
                count_good_evicts += len(np.where(next_hits >= BELADY_BOUNDARY_COE * CACHE_SIZE)[0])
                print("GDR:", count_good_evicts / count_evicts, count_good_evicts, count_evicts)
                print("LABELS", cache.reg.labeled_count)
            pbar.update(len(batch_embeds))

def test_freq_reg():
    DIM = 384
    DELTAS_COUNT = 8
    STREAM_SIZE = 100000
    CACHE_SIZE = 10000
    SAME_EMBED_DISTANCE = .5
    BATCH_SIZE = 10
    for dataset_name, embeds in load_embeds():
        reg = FreqReg(CACHE_SIZE, CACHE_SIZE, DELTAS_COUNT, SAME_EMBED_DISTANCE, DIM)
        embeds = reduce_dim(embeds, DIM, STREAM_SIZE)
        print(dataset_name)
        embeds = embeds[:STREAM_SIZE]
        embeds_covers = create_embeds_covers(embeds, SAME_EMBED_DISTANCE)
        total_mean = 0
        total_count = 0
        pbar = tqdm.tqdm(total=len(embeds), desc=f"Processing {dataset_name}...")
        for batch_embeds, i_embeds in yield_batches(embeds, BATCH_SIZE):
            pbar.update(len(batch_embeds))
            if reg.is_trained():
                predict = reg.predict(i_embeds, batch_embeds)
                actual = get_counts_hits(embeds_covers, i_embeds[0], i_embeds)
                actual = actual / (len(embeds) - np.array(i_embeds))
                mean = np.mean(np.abs(actual - predict))
                total_count += 1
                n = total_count
                total_mean = total_mean * (n-1)/n + mean / n
            reg.record_for_training(i_embeds, batch_embeds)
        print(total_mean)
        
    
if __name__ == "__main__":
    test_freq_reg()

                