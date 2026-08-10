from collections import defaultdict, deque
from caches.cache import Cache
import numpy as np

class Distribution(Cache):
    def __init__(self, same_embed_distance, dist_estimator):
        super().__init__(same_embed_distance)
        self.dist_estimator = dist_estimator

    def initialize(self, capacity: int, index):
        self.items = {}
        super().initialize(capacity, index)

    def cache(self, embeds, embeds_ids, count_nn=1, texts=[]):
        closest_dists, closest_ids = self.get_closest_stored_embeds(embeds, count_nn)
        hits_mask = closest_dists < self.same_embed_distance
        cache_hits = np.count_nonzero(hits_mask, axis=1)

        insert_embeds = []
        insert_embeds_ids = []
        evicted_items = []

        if len(insert_embeds) > 0:
            self.add_with_ids(insert_embeds, insert_embeds_ids)
        return cache_hits, evicted_items
