from heapdict import heapdict

import numpy as np
from caches.cache import Cache
from util.hill_climber import HillClimber
class LRFU(Cache):
    def __init__(self, same_embed_distance, decay_coe: float):
        super().__init__(same_embed_distance)
        self.decay_coe = decay_coe

    def initialize(self, capacity: int, index):
        self.items = heapdict()
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
            self.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        for _ in range(count_remove):
            evict_key, _ = self.items.popitem()
            evicted_items.append(evict_key)
        if evicted_items:
            self.remove_ids(np.array(evicted_items))
        self.time += len(embeds)
        return cache_hits, evicted_items

class HillClimbingLRFU(LRFU):
    def __init__(self, same_embed_distance, decay_coe: float, window_size: int):
        self.window_size = window_size
        super().__init__(same_embed_distance, decay_coe)

    def initialize(self, capacity: int, index):
        super().initialize(capacity, index)
        self.window_hits = 0
        self.window_counter = 0
        self.hill_climber = HillClimber(max_val=1, initial_value=self.decay_coe)

    def request(self, embeds, embeds_ids, count_nn=1, texts = []):
        cache_hits, evicted_items = super().request(embeds, embeds_ids, count_nn)
        self.window_counter += len(embeds)
        self.window_hits += np.count_nonzero(cache_hits)
        if self.window_counter >= self.window_size:
            score = self.window_hits / self.window_counter
            self.hill_climber.update(score)
            self.decay_coe = self.hill_climber.propose()
            print("Score", score, "Lambda", self.decay_coe)
            self.window_hits = 0
            self.window_counter = 0
        return cache_hits, evicted_items
        
class DeltaLRFU(LRFU):
    def __init__(self, same_embed_distance, decay_coe: float, lambda_delta: float):
        self.lambda_delta = lambda_delta
        super().__init__(same_embed_distance, decay_coe)

    def request(self, embeds, embeds_ids, count_nn=1):
        self.decay_coe += self.lambda_delta * len(embeds) 
        return super().request(embeds, embeds_ids, count_nn)