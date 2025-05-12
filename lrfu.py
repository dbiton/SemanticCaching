from collections import OrderedDict

import numpy as np
from cache import Cache

class LRFU(Cache):
    def __init__(self, same_embed_distance, decay_coe: float):
        super().__init__(same_embed_distance)
        self.decay_coe = decay_coe

    def initialize(self, capacity: int, index):
        self.items = OrderedDict()
        super().initialize(capacity, index)
        self.time = 0

    def _decayed_crf(self, crf, last_access_time):
        dt = self.time - last_access_time
        return crf * np.exp(-self.decay_coe * dt)

    def request(self, embeds, embeds_ids, count_nn=1):
        closest_dists, closest_ids = self.get_closest_stored_embeds(embeds, count_nn)
        mask = closest_dists < self.same_embed_distance
        cache_hits_indices = np.where(mask)
        cache_hits = np.count_nonzero(mask, axis=1)
        evicted_items = []
        additions_embeds = []
        additions_ids = []
        count_remove = max(0, (len(embeds) + self.size()) - self.capacity)
        
        for i_embed, i_nn in zip(*cache_hits_indices):
            nn_embed_id = closest_ids[i_embed][i_nn]
            crf, last = self.items[nn_embed_id]
            new_crf = 1 + self._decayed_crf(crf, last)
            self.items[nn_embed_id] = (new_crf, self.time)
        
        for embed_id, embed in zip(embeds_ids, embeds):
            self.items[embed_id] = (1.0, self.time)
            additions_embeds.append(embed)
            additions_ids.append(embed_id)
        
        if additions_ids:
            self.index.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        for _ in range(count_remove):
            evict_key = min(self.items.items(), key=lambda item: self._decayed_crf(item[1][0], item[1][1]))[0]
            del self.items[evict_key]
            evicted_items.append(evict_key)
        if evicted_items:
            self.index.remove_ids(np.array(evicted_items))
        self.time += len(embeds)
        return cache_hits, evicted_items