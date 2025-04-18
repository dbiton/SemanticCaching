import os
import pickle
import numpy as np
import faiss
import tqdm

from OPT import RLB_Reg, RelaxedLearnedOPT
from cache import LRU
from reduce_dim import reduce_dim

def load_embeds():
    filenames = {
        "Bing": "embeds_bing.pkl",
        #"StackOverflow": "embeds_so.pkl",
        #"WildChat": "embeds_chat.pkl"
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
    index = faiss.IndexFlatL2(d)
    index.add(embeds)
    threshold = same_embed_distance ** 2
    lims, _, indices = index.range_search(embeds, threshold)
    embeds_covers = []
    for i in range(n):
        i_covers = np.sort(np.array([j for j in indices[lims[i]:lims[i+1]] if j > i]))
        embeds_covers.append(i_covers)
    return np.array(embeds_covers, dtype=object)

def yield_batches(lst, k):
    for i in range(0, len(lst), k):
        yield lst[i:i + k], list(range(i, min(i+k, len(lst))))


def test_regressor():
    DIM = 384
    DELTAS_COUNT = 4
    STREAM_SIZE = 10000
    CACHE_SIZE = 1000
    SAME_EMBED_DISTANCE = 1.0
    BELADY_BOUNDARY_COE = 2.0
    BATCH_SIZE = 16
    reg = RLB_Reg(CACHE_SIZE, DELTAS_COUNT, SAME_EMBED_DISTANCE, BELADY_BOUNDARY_COE, DIM)
    for dataset_name, embeds in load_embeds():
        embeds = embeds[:STREAM_SIZE]
        embeds_covers = create_embeds_covers(embeds, SAME_EMBED_DISTANCE)
        for batch_embeds, i_embeds in yield_batches(embeds, BATCH_SIZE):
            print(i_embeds[-1])
            if reg.is_trained():
                predict = reg.predict(i_embeds, batch_embeds)
                actual = get_next_hits(embeds_covers, i_embeds[0], i_embeds)
                actual[np.isposinf(actual)] = reg.get_default_label()
                actual = np.log2(actual)
                print(abs(predict-actual).mean())
            reg.record_for_training(i_embeds, batch_embeds)
                
# GDR: 0.17
if __name__ == "__main__":
    test_regressor()

def test_policy():
    DIM = 384
    DELTAS_COUNT = 4
    STREAM_SIZE = 50000
    CACHE_SIZE = 5000
    BATCH_SIZE = 1
    COUNT_NN = 1
    SAME_EMBED_DISTANCE = 1.0
    BELADY_BOUNDARY_COE = 2.0
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
                next_hits = get_next_hits(embeds_covers, next_i_embed, evicted_embeds_ids)
                count_good_evicts += len(np.where(next_hits > next_i_embed + BELADY_BOUNDARY_COE * CACHE_SIZE)[0])
            pbar.update(len(batch_embeds))
            if next_i_embed % 1000 == 0 and count_evicts > 0:
                print("GDR:", count_good_evicts / count_evicts, count_good_evicts, count_evicts)
                