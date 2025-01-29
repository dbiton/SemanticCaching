import random
from collections import OrderedDict, defaultdict, deque
from typing import Tuple
from scipy.spatial import distance_matrix

import numpy as np

class Cache:
    def __init__(self, same_embed_distance: float):
        self.items = None
        self.capacity = None
        self.index = None
        self.same_embed_distance = same_embed_distance
    
    def initialize(self, capacity: int, index):
        self.capacity = capacity
        self.index = index
    
    def request(self, embed, embed_id):
        raise Exception("virtual method")

    def get_closest_stored_embed(self, embed):
        distances, neighbors = self.index.search(embed, 1)
        return neighbors[0][0], distances[0][0]
    
    def size(self):
        return len(self.items)

class Dummy(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        super().initialize(capacity, index)
            
    def request(self, embed, embed_id):
        return False, embed_id

class LFU(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        self.items = {}
        super().initialize(capacity, index)
            
    def request(self, embed, embed_id):
        closest_embed_id, closest_embed_distance = self.get_closest_stored_embed(embed)
        cache_hit = False
        evicted_item = None
        if closest_embed_id in self.items and closest_embed_distance < self.same_embed_distance:
            self.items[closest_embed_id] += 1
            cache_hit = True
        if self.size() >= self.capacity:
            removed_embed_id = min(self.items, key=self.items.get)
            del self.items[removed_embed_id]
            evicted_item = removed_embed_id
            self.index.remove_ids(np.array([removed_embed_id]))
        self.items[embed_id] = 1
        self.index.add_with_ids(embed, np.array([embed_id]))
        return cache_hit, evicted_item

class LRU(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        self.items = {}
        self.time_index = 0
        super().initialize(capacity, index)
            
    def request(self, embed, embed_id):
        self.time_index += 1
        closest_embed_id, closest_embed_distance = self.get_closest_stored_embed(embed)
        cache_hit = False
        evicted_item = None
        if closest_embed_id in self.items and closest_embed_distance < self.same_embed_distance:
            self.items[closest_embed_id] = self.time_index
            cache_hit = True
        if self.size() >= self.capacity:
            removed_embed_id = min(self.items, key=self.items.get)
            del self.items[removed_embed_id]
            evicted_item = removed_embed_id
            self.index.remove_ids(np.array([removed_embed_id]))
        self.items[embed_id] = self.time_index
        self.index.add_with_ids(embed, np.array([embed_id]))
        return cache_hit, evicted_item

class OPT(Cache):
    def __init__(self, same_embed_distance, embeds):
        super().__init__(same_embed_distance)
        self.embeds_distances = distance_matrix(embeds, embeds)
        self.embeds_covers = (self.embeds_distances < same_embed_distance).astype(int)
        tri_l = np.tril_indices_from(self.embeds_covers)
        self.embeds_covers[tri_l] = 0

    def initialize(self, capacity: int, index):
        self.items = set()
        super().initialize(capacity, index)
            
    def request(self, embed, embed_id) -> int:
        closest_embed_id, closest_embed_distance = self.get_closest_stored_embed(embed)
        cache_hit = False
        evicted_item = None
        if closest_embed_id in self.items and closest_embed_distance < self.same_embed_distance:
            cache_hit = True

        if self.size() < self.capacity:
            self.items.add(embed_id)
            self.index.add_with_ids(embed, np.array([embed_id]))
            return cache_hit, evicted_item
        
        scores = {}
        stored_embeds_indices = list(self.items) + [embed_id]
        stored_embeds_covers = self.embeds_covers[stored_embeds_indices]
        count_covers = stored_embeds_covers.sum(axis=0) # number of covers by stored embeds for each future embedding
        col_mask = (count_covers == 1)[embed_id:] # every future embedding with a single cover by a stored embedding 
        row_portion = self.embeds_covers[:, embed_id:] == 1
        scores_array = np.sum(row_portion & col_mask, axis=1)
        scores = {e: scores_array[e] for e in stored_embeds_indices}
        
        removed_embed_id = min(scores, key=scores.get)
        removed_item_future_hits = scores[removed_embed_id]
        if removed_item_future_hits < scores[embed_id]:
            self.items.remove(removed_embed_id)
            self.index.remove_ids(np.array([removed_embed_id]))
            self.items.add(embed_id)
            self.index.add_with_ids(embed, np.array([embed_id]))
            evicted_item = removed_embed_id
        return cache_hit, evicted_item

class OPT2(Cache):
    def __init__(self, same_embed_distance, embeds):
        super().__init__(same_embed_distance)
        self.embeds_distances = distance_matrix(embeds, embeds)
        self.embeds_covers = (self.embeds_distances < same_embed_distance).astype(int)
        tri_l = np.tril_indices_from(self.embeds_covers)
        self.embeds_covers[tri_l] = 0

    def initialize(self, capacity: int, index):
        self.items = set()
        super().initialize(capacity, index)
            
    def request(self, embed, embed_id) -> int:
        closest_embed_id, closest_embed_distance = self.get_closest_stored_embed(embed)
        cache_hit = False
        if closest_embed_id in self.items and closest_embed_distance < self.same_embed_distance:
            cache_hit = True

        if self.size() < self.capacity:
            self.items.add(embed_id)
            self.index.add_with_ids(embed, np.array([embed_id]))
            return cache_hit
        
        relevant_embeds_indices = sorted(list(self.items) + [embed_id])
        stored_embeds_covers = self.embeds_covers[relevant_embeds_indices]
        stored_embeds_cache_hits = [np.flatnonzero(row) for row in stored_embeds_covers]
        stored_embeds_next_cache_hit = [v[0] if len(v) > 0 else float('inf') for v in stored_embeds_cache_hits]
        scores = {e: stored_embeds_next_cache_hit[i_e] for (i_e, e) in enumerate(relevant_embeds_indices)}
        
        removed_embed_id = max(scores, key=scores.get)
        removed_item_future_hits = scores[removed_embed_id]
        if removed_item_future_hits < scores[embed_id]:
            self.items.remove(removed_embed_id)
            self.index.remove_ids(np.array([removed_embed_id]))
            self.items.add(embed_id)
            self.index.add_with_ids(embed, np.array([embed_id]))
        return cache_hit