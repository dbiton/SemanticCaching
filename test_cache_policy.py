import pickle
import numpy as np
import pytest
import faiss

from cache import OPT, LFU


def test_opt_two_clusters():
    count_region1 = 256
    count_region2 = 512
    np.random.seed(42)
    region1 = np.random.rand(count_region1, 2) * 10
    region2 = np.random.rand(count_region2, 2) * 10
    region2[:, 0] += 10
    embeds = np.vstack([region1, region2])
    np.random.shuffle(embeds)
    same_embed_distance = 1
    
    lfu = LFU(same_embed_distance)
    opt = OPT(same_embed_distance, embeds)
    
    index_lfu = faiss.IndexIDMap2(faiss.IndexFlatL2(2))
    index_opt = faiss.IndexIDMap2(faiss.IndexFlatL2(2))
    opt.initialize(32, index_opt)
    lfu.initialize(32, index_lfu)
    
    opt_cache_hits = 0
    lfu_cache_hits = 0
    for embed_id, embed in enumerate(embeds):
        opt_cache_hits += np.sum(opt.request(embed.reshape(1, -1), [embed_id])[0])
        lfu_cache_hits += np.sum(lfu.request(embed.reshape(1, -1), [embed_id])[0])
    print("LFU", lfu_cache_hits, "OPT", opt_cache_hits)
    assert lfu_cache_hits <= opt_cache_hits

from scipy.spatial import distance_matrix

def test_wildchat():
    embeds_path = "datasets/embeds_chat.pkl"
    with open(embeds_path, "rb") as f:
        embeds = pickle.load(f)
    cache_size = 128
    stream_size = 1024
    embeds = embeds[:stream_size]
    
    same_embed_distance = 0.5
    
    lfu = LFU(same_embed_distance)
    opt = OPT(same_embed_distance, embeds)
    
    index_lfu = faiss.IndexIDMap2(faiss.IndexFlatL2(384))
    index_opt = faiss.IndexIDMap2(faiss.IndexFlatL2(384))
    
    opt.initialize(cache_size, index_opt)
    lfu.initialize(cache_size, index_lfu)
    
    opt_cache_hits = 0
    lfu_cache_hits = 0
    for embed_id, embed in enumerate(embeds):
        opt_cache_hits += np.sum(opt.request(embed.reshape(1, -1), [embed_id])[0])
        lfu_cache_hits += np.sum(lfu.request(embed.reshape(1, -1), [embed_id])[0])
    print("LFU", lfu_cache_hits, "OPT", opt_cache_hits)
    assert lfu_cache_hits <= opt_cache_hits

if __name__ == "__main__":
    pytest.main()
