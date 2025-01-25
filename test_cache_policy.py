import numpy as np
import pytest

from cache_policy import OPT

def test_opt_two_clusters():
    count_region1 = 100
    count_region2 = 200
    region1 = np.random.rand(count_region1, 2)
    region2 = np.random.rand(count_region2, 2)
    region2[:, 0] += 10
    embeds = np.vstack([region1, region2])
    np.random.shuffle(embeds)

    same_embed_distance = 2
    
    p = OPT(embeds, same_embed_distance)
    p.set_size(2)
    cache = {}
    cache_hits = 0
    for i_embed, embed in enumerate(embeds):
        if len(cache) > 0:
            stored_embeds = np.array(list(cache.values()))
            distances = np.linalg.norm(stored_embeds - embed, axis=1)
            if distances.min() < same_embed_distance:
                cache_hits += 1
        id_remove, add_id = p.log_access(i_embed, embed, None)
        if id_remove != -1:
            print("pop", id_remove, embeds[id_remove])
            cache.pop(id_remove)
        if add_id:
            print("push", i_embed, embed)
            cache[i_embed] = embed
    expected_cache_hits = count_region1 + count_region2 - 2
    assert expected_cache_hits == cache_hits
    
if __name__ == "__main__":
    pytest.main()
