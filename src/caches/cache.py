import heapq
import random
from collections import OrderedDict, defaultdict, deque
import heapq
import numpy as np
from scipy.spatial import distance_matrix
import faiss

from util.surprisal import calculate_surprisal
from util.online_clusters import OnlineClusters
from typing import List

i = 0
class Cache:
    def __init__(self, same_embed_distance: float):
        self.items = None  # our internal cache storage
        self.capacity = None
        self.index = None  # an external “index” object
        self.same_embed_distance = same_embed_distance

    def initialize(self, capacity: int, index):
        self.capacity = capacity
        self.index = index

    def cache(self, embeds, embeds_ids, count_nn=1, texts=[]):
        raise NotImplementedError("virtual method")

    def add_with_ids(self, embeds: list[np.ndarray], embeds_ids: List[str]):
        self.index.add_with_ids(np.array(embeds).astype('float32'), np.array(embeds_ids).astype('int64'))
    
    def remove_ids(self, embeds_ids: List[str]):
        self.index.remove_ids(np.array(embeds_ids).astype('int64'))
    
    def get_closest_stored_embeds(self, embeds, count_nn=1):
        dists, ids = self.index.search(embeds, count_nn)
        dists = np.sqrt(dists)
        return dists, ids
    
    def get_cache_hits(self, embeds, count_nn=1) -> List[List[int]]:
        dists, ids = self.get_closest_stored_embeds(embeds, count_nn)
        cache_hits_ids = []
        cache_hits_dists = []
        for embed_hits_dists, embed_hits_ids in zip(dists, ids):
            cache_hits_dists.append([hdis for hdis in embed_hits_dists if hdis <= self.same_embed_distance])
            cache_hits_ids.append([hid for (hid, hdis) in zip(embed_hits_ids, embed_hits_dists) if hdis <= self.same_embed_distance])
        return cache_hits_ids, cache_hits_dists
    
    def get_in_range_stored_embeds(self, embeds, radius):
        radius_squared = radius ** 2
        lims, dist2, ids = self.index.range_search(embeds, radius_squared)
        dists = np.sqrt(dist2)
        formatted_dists = []
        formatted_ids = []
        start_index = 0
        for lim in lims[1:]:
            end_index = lim
            formatted_dists.append(dists[start_index:end_index])
            formatted_ids.append(ids[start_index:end_index])
            start_index = end_index
        return np.array(formatted_dists, dtype=object), np.array(formatted_ids, dtype=object)
    
    def size(self):
        return len(self.items)


class Dummy(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        super().initialize(capacity, index)

    def cache(self, embeds, embeds_ids, count_nn=1, texts=[]):
        shape = (len(embeds),)
        return np.zeros(shape, dtype=int), embeds_ids

class FIFO(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        self.items = OrderedDict()
        super().initialize(capacity, index)

    def cache(self, embeds, embeds_ids, count_nn=1, texts=[]):
        closest_dists, _ = self.get_closest_stored_embeds(embeds, count_nn)
        mask = closest_dists < self.same_embed_distance
        cache_hits = np.count_nonzero(mask, axis=1)

        evicted_items = []
        additions_embeds, additions_ids = [], []
        count_remove = max(0, (len(embeds) + self.size()) - self.capacity)

        for embed_id, embed in zip(embeds_ids, embeds):
            self.items[embed_id] = None
            additions_embeds.append(embed)
            additions_ids.append(embed_id)

        if additions_ids:
            self.add_with_ids(np.array(additions_embeds), np.array(additions_ids))

        for _ in range(count_remove):
            victim_id, _ = self.items.popitem(last=False)
            evicted_items.append(victim_id)

        if evicted_items:
            self.remove_ids(np.array(evicted_items))

        return cache_hits, evicted_items


class LIFO(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        self.items = OrderedDict()
        super().initialize(capacity, index)

    def cache(self, embeds, embeds_ids, count_nn=1, texts=[]):
        closest_dists, _ = self.get_closest_stored_embeds(embeds, count_nn)
        mask = closest_dists < self.same_embed_distance
        cache_hits = np.count_nonzero(mask, axis=1)

        evicted_items = []
        additions_embeds, additions_ids = [], []
        count_remove = max(0, (len(embeds) + self.size()) - self.capacity)

        for embed_id, embed in zip(embeds_ids, embeds):
            self.items[embed_id] = None
            additions_embeds.append(embed)
            additions_ids.append(embed_id)

        if additions_ids:
            self.add_with_ids(np.array(additions_embeds), np.array(additions_ids))

        for _ in range(count_remove):
            victim_id, _ = self.items.popitem(last=True)
            evicted_items.append(victim_id)

        if evicted_items:
            self.remove_ids(np.array(evicted_items))

        return cache_hits, evicted_items
class RR(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        # Use a plain dict mapping embed_id -> usage count.
        self.items = []
        super().initialize(capacity, index)

    def cache(self, embeds, embeds_ids, count_nn=1, texts=[]):
        assert(len(embeds) < self.capacity)
        closest_dists, closest_ids = self.get_closest_stored_embeds(embeds, count_nn)
        cache_hits = np.sum(closest_dists < self.same_embed_distance, axis=1)
        count_remove = max(0, (len(embeds) + self.size()) - self.capacity)
        removed_indices = sorted(random.sample(range(len(self.items)), count_remove), reverse=True)
        removals = np.array([self.items[i] for i in removed_indices])
        for i in removed_indices:
            self.items.pop(i)
        if len(removals) > 0:
            self.remove_ids(removals)
        self.add_with_ids(embeds, embeds_ids)
        self.items += embeds_ids
        return cache_hits, removals

class ClusterLFU(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        # Use a dict mapping embed_id -> frequency (access count)
        self.items = {}
        self.clusters_counter = {}
        self.online_clusters = OnlineClusters(self.same_embed_distance, 384)
        super().initialize(capacity, index)

    def cache(self, embeds, embeds_ids, count_nn=1, texts=[]):
        closest_dists, closest_ids = self.get_closest_stored_embeds(embeds, count_nn)
        mask = closest_dists < self.same_embed_distance
        cache_hits_indices = np.where(mask)
        cache_hits = np.count_nonzero(mask, axis=1)
        evicted_items = []
        additions_embeds = []
        additions_ids = []
        for i_embed, i_nn in zip(*cache_hits_indices):
            cand = closest_ids[i_embed][i_nn]
            cluster_id = self.online_clusters.get_cluster(cand)
            self.clusters_counter[cluster_id] += 1

        needed_space = len(embeds)
        current_size = self.size()
        count_remove = max(0, (current_size + needed_space) - self.capacity)
        for _ in range(count_remove):
            cluster_id = min(self.clusters_counter, key=self.clusters_counter.get)
            embed_id, cluster_emptied = self.online_clusters.pop_vector(cluster_id)
            if cluster_emptied:
                self.clusters_counter.pop(cluster_id)
            self.items.pop(embed_id, None)
            evicted_items.append(embed_id)

        for embed_id, embed in zip(embeds_ids, embeds):
            cluster_id = self.online_clusters.add_vector(embed, embed_id)
            self.clusters_counter[cluster_id] = 1
            self.items[embed_id] = cluster_id
            additions_embeds.append(embed)
            additions_ids.append(embed_id)
        if evicted_items:
            self.remove_ids(np.array(evicted_items))
        if additions_ids:
            self.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        return cache_hits, evicted_items

class LFU(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        # Use a dict mapping embed_id -> frequency (access count)
        self.items = {}
        super().initialize(capacity, index)

    def cache(self, embeds, embeds_ids, count_nn=1, texts=[]):
        closest_dists, closest_ids = self.get_closest_stored_embeds(embeds, count_nn)
        hit_ids = []
        mask = closest_dists < self.same_embed_distance
        cache_hits_indices = np.where(mask)
        cache_hits = np.count_nonzero(mask, axis=1)
        evicted_items = []
        additions_embeds = []
        additions_ids = []
        count_remove = max(0, (len(embeds) + self.size()) - self.capacity)
        for i_embed, i_nn in zip(*cache_hits_indices):
            cand = closest_ids[i_embed][i_nn]
            self.items[cand] += 1
        needed_space = len(embeds)
        current_size = self.size()
        count_remove = max(0, (current_size + needed_space) - self.capacity)
        if count_remove > 0:
            least_used = heapq.nsmallest(count_remove, self.items.items(), key=lambda x: x[1])
            for embed_id, _ in least_used:
                del self.items[embed_id]
                evicted_items.append(embed_id)
        for embed_id, embed in zip(embeds_ids, embeds):
            self.items[embed_id] = 0
            additions_embeds.append(embed)
            additions_ids.append(embed_id)
        if len(evicted_items) > 0:
            self.remove_ids(evicted_items)
        if additions_ids:
            self.add_with_ids(additions_embeds, additions_ids)
        return cache_hits, evicted_items

class MissLFU(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        # Use a dict mapping embed_id -> frequency (access count)
        self.items = {}
        super().initialize(capacity, index)

    def cache(self, embeds, embeds_ids, count_nn=1, texts=[]):
        closest_dists, closest_ids = self.get_closest_stored_embeds(embeds, count_nn)
        mask = closest_dists < self.same_embed_distance
        cache_hits_indices = np.where(mask)
        cache_hits = np.count_nonzero(mask, axis=1)
        evicted_items = []
        additions_embeds = []
        additions_ids = []
        embeds_ids_cand = set(embeds_ids)
        for i_embed, i_nn in zip(*cache_hits_indices):
            cand = closest_ids[i_embed][i_nn]
            embed_id = embeds_ids[i_embed]
            self.items[cand] += 1
            embeds_ids_cand.discard(embed_id)
        needed_space = len(embeds_ids_cand)
        current_size = self.size()
        count_remove = max(0, (current_size + needed_space) - self.capacity)
        if count_remove > 0:
            least_used = heapq.nsmallest(count_remove, self.items.items(), key=lambda x: x[1])
            for embed_id, _ in least_used:
                del self.items[embed_id]
                evicted_items.append(embed_id)
        for embed_id, embed in zip(embeds_ids, embeds):
            if embed_id not in embeds_ids_cand:
                continue
            self.items[embed_id] = 0
            additions_embeds.append(embed)
            additions_ids.append(embed_id)
        if len(evicted_items) > 0:
            self.remove_ids(evicted_items)
        if additions_ids:
            self.add_with_ids(additions_embeds, additions_ids)
        return cache_hits, evicted_items

class SurprisalLFU(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        # Use a dict mapping embed_id -> frequency (access count)
        self.items = {}
        super().initialize(capacity, index)

    def cache(self, embeds, embeds_ids, count_nn=1, texts=[]):
        closest_dists, closest_ids = self.get_closest_stored_embeds(embeds, count_nn)
        mask = closest_dists < self.same_embed_distance
        cache_hits_indices = np.where(mask)
        cache_hits = np.count_nonzero(mask, axis=1)
        evicted_items = []
        additions_embeds = []
        additions_ids = []
        count_remove = max(0, (len(embeds) + self.size()) - self.capacity)
        for i_embed, i_nn in zip(*cache_hits_indices):
            cand = closest_ids[i_embed][i_nn]
            self.items[cand][0] += 1

        needed_space = len(embeds)
        current_size = self.size()
        count_remove = max(0, (current_size + needed_space) - self.capacity)
        if count_remove > 0:
            least_used = heapq.nsmallest(count_remove, self.items.items(), key=lambda x: x[1])
            for embed_id, _ in least_used:
                del self.items[embed_id]
                evicted_items.append(embed_id)

        for embed_id, embed, embed_text in zip(embeds_ids, embeds, texts):
            self.items[embed_id] = [0, -calculate_surprisal(embed_text)]
            additions_embeds.append(embed)
            additions_ids.append(embed_id)
        if evicted_items:
            self.remove_ids(np.array(evicted_items))
        if additions_ids:
            self.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        return cache_hits, evicted_items

class SimpleCache(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def get_score(self, embed, embed_id: int, embed_text: str) -> float:
        return 0
    
    def initialize(self, capacity: int, index):
        self.items = {}
        super().initialize(capacity, index)

    def cache(self, embeds, embeds_ids, count_nn=1, texts=[]):
        closest_dists, _ = self.get_closest_stored_embeds(embeds, count_nn)
        mask = closest_dists < self.same_embed_distance
        cache_hits = np.count_nonzero(mask, axis=1)
        evicted_items = []
        additions_embeds = []
        additions_ids = []
        count_remove = max(0, (len(embeds) + self.size()) - self.capacity)

        needed_space = len(embeds)
        current_size = self.size()
        count_remove = max(0, (current_size + needed_space) - self.capacity)
        if count_remove > 0:
            least_used = heapq.nsmallest(count_remove, self.items.items(), key=lambda x: x[1])
            for embed_id, _ in least_used:
                del self.items[embed_id]
                evicted_items.append(embed_id)

        for embed_id, embed, text in zip(embeds_ids, embeds, texts):
            self.items[embed_id] = self.get_score(embed, embed_id, text)
            additions_embeds.append(embed)
            additions_ids.append(embed_id)
        if evicted_items:
            self.remove_ids(np.array(evicted_items))
        if additions_ids:
            self.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        return cache_hits, evicted_items

class CountChars(SimpleCache):
    def get_score(self, embed, embed_id, embed_text):
        return -len(embed_text)
    
class CountWords(SimpleCache):
    def get_score(self, embed, embed_id, embed_text):
        return -len(embed_text.split())
    
class Surprisal(SimpleCache):
    def get_score(self, embed, embed_id, embed_text):
        return -calculate_surprisal(embed_text)
    
class SphereQueryLFU(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        # Use a dict mapping embed_id -> frequency (access count)
        self.items = {}
        super().initialize(capacity, index)
    
    def cache(self, embeds, embeds_ids, count_nn=1, texts=[]):
        closest_dists, closest_ids = self.get_in_range_stored_embeds(embeds, self.same_embed_distance)
        mask = closest_dists < self.same_embed_distance
        cache_hits_indices = np.where(mask)
        cache_hits = np.count_nonzero(mask, axis=1)
        evicted_items = []
        additions_embeds = []
        additions_ids = []
        count_remove = max(0, (len(embeds) + self.size()) - self.capacity)
        for i_embed, i_nn in zip(*cache_hits_indices):
            embed_count_hits = len(closest_ids[i_embed])
            cand = closest_ids[i_embed][i_nn]
            self.items[cand] += 1 / embed_count_hits

        needed_space = len(embeds)
        current_size = self.size()
        count_remove = max(0, (current_size + needed_space) - self.capacity)
        if count_remove > 0:
            least_used = heapq.nsmallest(count_remove, self.items.items(), key=lambda x: x[1])
            for embed_id, _ in least_used:
                del self.items[embed_id]
                evicted_items.append(embed_id)

        for embed_id, embed in zip(embeds_ids, embeds):
            self.items[embed_id] = 0
            additions_embeds.append(embed)
            additions_ids.append(embed_id)
        if evicted_items:
            self.remove_ids(np.array(evicted_items))
        if additions_ids:
            self.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        return cache_hits, evicted_items

class DistanceLFU(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        self.items = {}
        super().initialize(capacity, index)
    
    def cache(self, embeds, embeds_ids, count_nn=1, texts=[]):
        closest_dists, closest_ids = self.get_closest_stored_embeds(embeds, count_nn)
        mask = closest_dists < self.same_embed_distance
        cache_hits_indices = np.where(mask)
        cache_hits = np.sum(mask, axis=1)
        evicted_items = []
        additions_embeds = []
        additions_ids = []
        count_remove = max(0, (len(embeds) + self.size()) - self.capacity)
        for i_embed, i_nn in zip(*cache_hits_indices):
            cand = closest_ids[i_embed][i_nn]
            cand_dist = closest_dists[i_embed][i_nn]
            score = 1 - cand_dist / self.same_embed_distance
            self.items[cand] += score
        for _ in range(count_remove):
            evicted_embed_id = min(self.items, key=self.items.get)
            del self.items[evicted_embed_id]
            evicted_items.append(evicted_embed_id)
        for embed_id, embed in zip(embeds_ids, embeds):
            self.items[embed_id] = 0
            additions_embeds.append(embed)
            additions_ids.append(embed_id)
        if evicted_items:
            self.remove_ids(np.array(evicted_items))
        if additions_ids:
            self.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        return cache_hits, evicted_items

class DynamicAgingLFU(Cache):
    """
    LFUDA (LFU with Dynamic Aging):
      - Global age L starts at 0.
      - Each item keeps a 'key' that effectively equals (reference_count + L_at_last_touch).
      - On hit: key += 1
      - On insert: key = L + 1
      - Evict min-key; set L = evicted_key (dynamic aging).
      - Invariant: L <= min_key_in_cache.
    """
    def __init__(self, same_embed_distance, half_life: int = None):
        # 'half_life' kept only for API compatibility. LFUDA does not use exponential decay.
        super().__init__(same_embed_distance)
        self.L = 0.0
        self.half_life = half_life

    def initialize(self, capacity: int, index):
        self.items = {}   # embed_id -> key (float)
        self.L = 0.0
        super().initialize(capacity, index)

    def cache(self, embeds, embeds_ids, count_nn=1, texts=[]):
        # 1) Query current index to find hits
        closest_dists, closest_ids = self.get_closest_stored_embeds(embeds, count_nn)
        mask = closest_dists < self.same_embed_distance
        cache_hits_indices = np.where(mask)
        cache_hits = np.count_nonzero(mask, axis=1)

        evicted_items = []
        additions_embeds = []
        additions_ids = []

        # 2) Re-references: increment key by 1 (key already included L when last touched)
        for i_embed, i_nn in zip(*cache_hits_indices):
            cand = closest_ids[i_embed][i_nn]
            if cand in self.items:         # defensive guard
                self.items[cand] += 1.0

        # 3) Evict enough to make room for all incoming inserts
        needed_space = len(embeds)
        current_size = self.size()
        count_remove = max(0, (current_size + needed_space) - self.capacity)
        for _ in range(count_remove):
            evict_id, evict_key = min(self.items.items(), key=lambda kv: kv[1])
            self.L = float(evict_key)      # dynamic aging; maintains L <= min_key_in_cache
            del self.items[evict_id]
            evicted_items.append(evict_id)

        # 4) Admit all new objects with key = L + 1 (adds cache-age factor on admission)
        for embed_id, embed in zip(embeds_ids, embeds):
            self.items[embed_id] = self.L + 1.0
            additions_embeds.append(embed)
            additions_ids.append(embed_id)

        # 5) Sync to underlying vector index
        if evicted_items:
            self.remove_ids(np.array(evicted_items))
        if additions_ids:
            self.add_with_ids(np.array(additions_embeds), np.array(additions_ids))

        return cache_hits, evicted_items

class LRU(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        self.items = OrderedDict()
        super().initialize(capacity, index)

    def cache(self, embeds, embeds_ids, count_nn=1, texts=[]):
        closest_dists, closest_ids = self.get_closest_stored_embeds(embeds, count_nn)
        mask = closest_dists < self.same_embed_distance
        cache_hits_indices = np.where(mask)
        cache_hits = np.count_nonzero(mask, axis=1)
        evicted_items = []
        additions_embeds = []
        additions_ids = []
        count_remove = max(0, (len(embeds) + self.size()) - self.capacity)
        for i_embed, i_nn in zip(*cache_hits_indices):
            nn = closest_ids[i_embed][i_nn]
            self.items.move_to_end(nn)
        for embed_id, embed in zip(embeds_ids, embeds):
            self.items[embed_id] = None
            additions_embeds.append(embed)
            additions_ids.append(embed_id)
        if additions_ids:
            self.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        for _ in range(count_remove):
            embed_id, _ = self.items.popitem(last = False)
            evicted_items.append(embed_id)
        if evicted_items:
            self.remove_ids(np.array(evicted_items))
        return cache_hits, evicted_items

class RAP(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        self.items = {}
        super().initialize(capacity, index)

    def cache(self, embeds, embeds_ids, count_nn=1, texts=[]):
        closest_dists, closest_ids = self.get_closest_stored_embeds(embeds, count_nn)
        mask = closest_dists < self.same_embed_distance
        cache_hits_indices = np.where(mask)
        cache_hits = np.count_nonzero(mask, axis=1)
        removed_items = []
        rejected_items = []
        additions_embeds = []
        additions_ids = []
        for i_embed, i_nn in zip(*cache_hits_indices):
            cand = closest_ids[i_embed][i_nn]
            self.items[cand] += 1
        for embed_id, embed in zip(embeds_ids, embeds):
            if self.capacity <= self.size():
                cand_embed_id = min(self.items, key=self.items.get)
                cand_hits = self.items.get(cand_embed_id)
                thresh = 1.0 / (cand_hits + 1)
                if random.random() >= thresh:
                    rejected_items.append(embed_id)
                    continue
                else:
                    removed_items.append(cand_embed_id)
                    self.items.pop(cand_embed_id)
            additions_embeds.append(embed)
            additions_ids.append(embed_id)
        if removed_items:
            self.remove_ids(np.array(removed_items))
        if additions_ids:
            self.items.update({eid: 0 for eid in additions_ids})
            self.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        return cache_hits, removed_items + rejected_items

class FixedRadius(Cache):
    def __init__(self, same_embed_distance, radius):
        super().__init__(same_embed_distance)
        self.radius = radius

    def initialize(self, capacity: int, index):
        self.items = {}
        super().initialize(capacity, index)

    def cache(self, embeds, embeds_ids, count_nn=1, texts=[]):
        closest_dists, closest_ids = self.get_closest_stored_embeds(embeds, count_nn)
        cache_hits = np.sum(closest_dists < self.same_embed_distance, axis=1)
        evicted_items = []
        additions = []
        for i, embed_id in enumerate(embeds_ids):
            embed_closest_ids = closest_ids[i]
            embed_closest_distances = closest_dists[i]
            size_neigh = np.sum(embed_closest_distances < self.radius)
            for i_nn, (nn_embed_id, nn_embed_distance) in enumerate(zip(embed_closest_ids, embed_closest_distances)):
                if nn_embed_distance <= self.radius:
                    distance_factor = nn_embed_distance / self.radius
                    self.items[nn_embed_id] = self.items[nn_embed_id] * distance_factor + size_neigh * (1 - distance_factor)
            if np.sum(cache_hits[i]) == 0 or self.size() < self.capacity:
                additions.append((embed_id, embeds[i], size_neigh))
            else:
                evicted_items.append(embed_id)
        count_removed = (len(additions) + self.size()) - self.capacity        
        if count_removed > 0:
            removed_items = heapq.nsmallest(count_removed, self.items, key=self.items.get)
            for k in removed_items:
                del self.items[k]
            self.remove_ids(np.array(removed_items))
            evicted_items += removed_items
        if additions:
            additions_embeds = [v for (_, v, _) in additions]
            additions_ids = [v for (v, _, _) in additions]
            for (embed_id, _, size_neigh) in additions:
                self.items[embed_id] = size_neigh
            self.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        return cache_hits, evicted_items

class PCA(Cache):
    def __init__(self, same_embed_distance, pca_dim = 9, cluster_diameter = 1.0):
        super().__init__(same_embed_distance)
        self.pca_dim = pca_dim
        self.train_counter = 0
        self.cluster_diameter = cluster_diameter
        self.pca = None

    def train_pca(self):
        n_embeds = self.index.ntotal
        dim_embeds = self.index.d
        embeds = self.index.index.reconstruct_n(0, n_embeds)
        embeds_ids = faiss.vector_to_array(self.index.id_map)
        self.pca = faiss.PCAMatrix(dim_embeds, self.pca_dim)
        self.pca.train(embeds)
        assert self.pca.is_trained
        embeds_pca = self.pca.apply(embeds)
        self.clusters = defaultdict(list)
        for embed_id, embed_pca in zip(embeds_ids, embeds_pca):
            cluster_id = self.embed_pca_to_cluster_id(embed_pca)
            self.clusters[cluster_id].append(embed_id)
        
    def embeds_to_clusters_ids(self, embeds):
        if not self.pca:
            return [-1 for _ in range(len(embeds))]
        embeds_pca = self.pca.apply(np.array(embeds))
        return [self.embed_pca_to_cluster_id(embed_pca) for embed_pca in embeds_pca]          
    
    def embed_pca_to_cluster_id(self, embed_pca):
        unit_hypercube_diameter = np.sqrt(self.pca_dim)
        hypercube_side_length = self.cluster_diameter / unit_hypercube_diameter
        cluster_id = np.round(embed_pca / hypercube_side_length).astype(int)
        return tuple(cluster_id)
    
    def initialize(self, capacity: int, index):
        self.items = {}
        super().initialize(capacity, index)

    def cache(self, embeds, embeds_ids, count_nn=1, texts=[]):
        self.train_counter += len(embeds)
        if self.train_counter >= self.capacity:
            self.train_counter = 0
            self.train_pca()
        closest_dists, closest_ids = self.get_closest_stored_embeds(embeds, count_nn)
        cache_hits = np.sum(closest_dists < self.same_embed_distance, axis=1)
        evicted_items = []
        additions = []
        for i, embed_id in enumerate(embeds_ids):
            if np.sum(cache_hits[i]) == 0 or self.size() < self.capacity:
                additions.append((embed_id, embeds[i]))
            else:
                evicted_items.append(embed_id)
        count_removed = (len(additions) + self.size()) - self.capacity        
        if count_removed > 0:
            for _ in range(count_removed):
                smallest_cluster = min(self.clusters, key=lambda k: len(self.clusters[k]))
                k = self.clusters[smallest_cluster].pop()
                if len(self.clusters[smallest_cluster]) == 0:
                    del self.clusters[smallest_cluster]
                del self.items[k]
                evicted_items.append(k)
            self.remove_ids(np.array(evicted_items))
        if additions:
            additions_embeds = [v for (_, v) in additions]
            additions_ids = [v for (v, _) in additions]
            additions_clusters_ids = self.embeds_to_clusters_ids(additions_embeds)
            for (embed_id, embed), cluster_id in zip(additions, additions_clusters_ids):
                self.items[embed_id] = cluster_id
            self.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        return cache_hits, evicted_items

class BetterTinyLFU(Cache):
    def __init__(self, same_embed_distance, window_ratio=16):
        super().__init__(same_embed_distance)
        self.window_ratio = window_ratio
    
    def initialize(self, capacity: int, index):
        self.decay_factor = 0.5
        self.sample_size = int(self.window_ratio * capacity)
        self.freq_sketch = defaultdict(int)
        self.capacity = capacity
        self.index = index
        self.index_freq = faiss.IndexIDMap2(faiss.IndexFlatL2(index.d))
        self.items = OrderedDict()

    def decay_frequencies(self):
        evicted_items = []
        for key in list(self.freq_sketch.keys()):
            self.freq_sketch[key] = self.freq_sketch[key] * self.decay_factor
            if self.freq_sketch[key] < 1:
                del self.freq_sketch[key]
                evicted_items.append(key)
        self.index_freq.remove_ids(np.array(evicted_items))
        return evicted_items

    def get_in_range_freq_embeds(self, embeds, radius):
        radius_squared = radius ** 2
        lims, dist2, ids = self.index_freq.range_search(embeds, radius_squared)
        dists = np.sqrt(dist2)
        formatted_dists = []
        formatted_ids = []
        start_index = 0
        for lim in lims[1:]:
            end_index = lim
            formatted_dists.append(dists[start_index:end_index])
            formatted_ids.append(ids[start_index:end_index])
            start_index = end_index
        return np.array(formatted_dists, dtype=object), np.array(formatted_ids, dtype=object)
    
    def update_frequencies(self, embeds, embeds_ids):
        _, closest_ids = self.get_in_range_freq_embeds(embeds, self.same_embed_distance)
        embed_frequencies = []
        for embed_id, embed_closest_ids in zip(embeds_ids, closest_ids):
            freq = 0
            for embed_neigh_id in embed_closest_ids:
                embed_count_hits = len(embed_closest_ids)
                new_freq = min(self.freq_sketch[embed_neigh_id] + 1 / embed_count_hits, self.window_ratio)
                self.freq_sketch[embed_neigh_id] = new_freq
                self.items.move_to_end(embed_neigh_id)
                freq += self.freq_sketch[embed_neigh_id]
            embed_frequencies.append(freq)
        return embed_frequencies
    
    def cache(self, embeds, embeds_ids, count_nn=1, texts=[]):
        closest_dists, closest_ids = self.get_in_range_stored_embeds(embeds, self.same_embed_distance)
        mask = closest_dists < self.same_embed_distance
        cache_hits = np.count_nonzero(mask, axis=1)
        removals = []
        additions_embeds = []
        additions_ids = []
        embed_frequencies = self.update_frequencies(embeds, embeds_ids)
        
        for embed_id, embed, embed_freq, embed_hits in zip(embeds_ids, embeds, embed_frequencies, cache_hits):
            if self.size() <= self.capacity:
                self.items[embed_id] = None
                additions_ids.append(embed_id)
                additions_embeds.append(embed)
            elif embed_hits == 0:
                victim_embed_id, _ = next(iter(self.items.items()))
                victim_freq = self.freq_sketch[victim_embed_id]
                if embed_freq > victim_freq:
                    self.items.popitem(last=False)
                    removals.append(victim_embed_id)
                    self.items[embed_id] = None
                    additions_ids.append(embed_id)
                    additions_embeds.append(embed)
                else:
                    removals.append(embed_id)
        
        while len(self.freq_sketch) >= self.sample_size:
            removals += self.decay_frequencies()
        
        if removals:
            self.remove_ids(np.array(removals))
        if additions_ids:
            self.add_with_ids(np.array(additions_embeds), np.array(additions_ids))

        return cache_hits, removals
    
class TinyLFU(Cache):
    def __init__(self, same_embed_distance,  window_ratio=16):
        super().__init__(same_embed_distance)
        self.window_ratio =  window_ratio

    def decay_frequencies(self):
        evicted_items = []
        for key in list(self.freq_sketch.keys()):
            self.freq_sketch[key] *= self.decay_factor
            if self.freq_sketch[key] < 1:
                del self.freq_sketch[key]
                evicted_items.append(key)
        return evicted_items
    
    def initialize(self, capacity: int, index):
        self.decay_factor = 0.5
        self.sample_size = int(self.window_ratio * capacity)
        self.freq_sketch = defaultdict(int)
        self.capacity = capacity
        self.index = index
        self.items = OrderedDict()

    def cache(self, embeds, embeds_ids, count_nn=1, texts=[]):
        closest_dists, closest_ids = self.get_in_range_stored_embeds(embeds, self.same_embed_distance)
        mask = closest_dists < self.same_embed_distance
        cache_hits = np.count_nonzero(mask, axis=1)
        removals = []
        additions_embeds = []
        additions_ids = []
        
        embed_frequencies = []
        for embed_id, embed_closest_ids in zip(embeds_ids, closest_ids):
            freq = 0
            for embed_neigh_id in embed_closest_ids:
                count_neighs = len(embed_closest_ids)
                new_freq = min(self.freq_sketch[embed_neigh_id] + 1 / count_neighs, self.window_ratio)
                self.freq_sketch[embed_neigh_id] = new_freq
                self.items.move_to_end(embed_neigh_id)
                freq += self.freq_sketch[embed_neigh_id]
            embed_frequencies.append(freq)

        for embed_id, embed, embed_freq, embed_hits in zip(embeds_ids, embeds, embed_frequencies, cache_hits):
            if self.size() <= self.capacity:
                self.items[embed_id] = None
                additions_ids.append(embed_id)
                additions_embeds.append(embed)
            elif embed_hits == 0:
                victim_embed_id, _ = next(iter(self.items.items()))
                victim_freq = self.freq_sketch[victim_embed_id]
                if embed_freq > victim_freq:
                    self.items.popitem(last=False)
                    removals.append(victim_embed_id)
                    self.items[embed_id] = None
                    additions_ids.append(embed_id)
                    additions_embeds.append(embed)
                else:
                    removals.append(embed_id)
        
        while len(self.freq_sketch) >= self.sample_size:
            removals += self.decay_frequencies()
        
        if removals:
            self.remove_ids(np.array(removals))
        if additions_ids:
            self.add_with_ids(np.array(additions_embeds), np.array(additions_ids))

        return cache_hits, removals


class SamplingMetaCache(Cache):
    def __init__(
        self,
        same_embed_distance: float,
        base_policies: list,
        sample_cache_ratio: float = 0.25,
        sample_prob: float = 1.0,
        min_sample_queries: int = 1000,
        switch_interval: int = 10000,
    ):
        """
        :param base_policies: list of Cache subclasses (e.g. [LRU, LFU, TinyLFU]).
                              These must have the same constructor signature as your
                              existing policies: cls(same_embed_distance).
        :param sample_cache_ratio: capacity of each shadow cache as a fraction
                                   of the main capacity (e.g. 0.25 -> 25%).
        :param sample_prob: probability that a given request is sent to a given
                            shadow cache (request sampling).
                            1.0 means all requests.
        :param min_sample_queries: minimal number of sampled queries per policy
                                   before we consider switching the active policy.
        :param switch_interval: number of *real* queries between possible switches.
        """
        super().__init__(same_embed_distance)
        assert 0.0 < sample_cache_ratio <= 1.0
        assert 0.0 < sample_prob <= 1.0

        self.base_policies = base_policies
        self.sample_cache_ratio = sample_cache_ratio
        self.sample_prob = sample_prob
        self.min_sample_queries = min_sample_queries
        self.switch_interval = switch_interval

        # Runtime state
        self.active_policy = None        # the production cache
        self.shadow_policies = []        # small sampled caches
        self.policy_names = [cls.__name__ for cls in base_policies]
        self.policy_stats = []           # list of dicts: {hits, queries}
        self.total_hits = 0
        self.total_queries = 0
        self.global_queries = 0

    def initialize(self, capacity: int, index):
        # We keep these for compatibility, but most logic is delegated.
        self.capacity = capacity
        self.index = index

        # 1) Active (production) policy: start with the first policy in the list.
        active_cls = self.base_policies[0]
        self.active_policy = active_cls(self.same_embed_distance)
        self.active_policy.initialize(capacity, index)
        self.items = self.active_policy.items
        
        # 2) Shadow policies: smaller independent caches + indices.
        self.shadow_policies = []
        self.policy_stats = []
        dim = index.dim  # FAISS dimension

        shadow_capacity = max(1, int(capacity * self.sample_cache_ratio))

        for cls in self.base_policies:
            shadow_index = faiss.IndexIDMap2(faiss.IndexFlatL2(dim))
            policy = cls(self.same_embed_distance)
            policy.initialize(shadow_capacity, shadow_index)
            self.shadow_policies.append(policy)
            self.policy_stats.append({"hits": 0, "queries": 0})

        # Reset stats
        self.total_hits = 0
        self.total_queries = 0
        self.global_queries = 0

    def size(self):
        if self.active_policy is None:
            return 0
        return self.active_policy.size()

    def get_policy_stats(self):
        result = {}
        for name, stats in zip(self.policy_names, self.policy_stats):
            q = stats["queries"]
            hr = (stats["hits"] / q) if q > 0 else 0.0
            result[name] = {
                "hits": stats["hits"],
                "queries": q,
                "hit_rate": hr,
            }
        return result

    def get_best_policy(self):
        stats = self.get_policy_stats()
        if not stats:
            return None, None
        best_name = None
        best_hr = -1.0
        for name, info in stats.items():
            if info["queries"] == 0:
                continue
            if info["hit_rate"] > best_hr:
                best_hr = info["hit_rate"]
                best_name = name
        if best_name is None:
            return None, None
        return best_name, stats[best_name]

    def get_overall_hit_rate(self):
        if self.total_queries == 0:
            return 0.0
        return self.total_hits / self.total_queries

    # ---------- Internal: switching active policy ----------

    def _maybe_switch_active_policy(self):
        """
        Optionally switch the active policy to the best-performing one.
        This is called periodically from cache().

        NOTE: For simplicity, when switching policies we FLUSH the real cache:
              - we reset the FAISS index
              - we reinitialize the new active policy as empty

        This keeps metadata consistent at the cost of losing current contents.
        """
        # Only consider switching every 'switch_interval' queries.
        if self.global_queries == 0:
            return
        if self.global_queries % self.switch_interval != 0:
            return

        # Require each policy to have seen at least min_sample_queries.
        for s in self.policy_stats:
            if s["queries"] < self.min_sample_queries:
                return

        # Choose best by hit-rate.
        best_name, best_info = self.get_best_policy()
        if best_name is None:
            return

        # If current active policy already matches best, do nothing.
        if type(self.active_policy).__name__ == best_name:
            return

        # Find class for best policy.
        best_cls = None
        for cls in self.base_policies:
            if cls.__name__ == best_name:
                best_cls = cls
                break
        if best_cls is None:
            return  # shouldn't happen

        # Flush real cache and reinitialize it with the best class.
        # We reset the FAISS index; content is lost but consistency is preserved.
        if self.index is not None:
            self.remove_ids(list(self.items))

        new_policy = best_cls(self.same_embed_distance)
        new_policy.initialize(self.capacity, self.index)
        self.active_policy = new_policy
        self.items = self.active_policy.items

        # We DO NOT reset policy_stats here; they continue accumulating.

    # ---------- Main API ----------

    def cache(self, embeds, embeds_ids, count_nn=1, texts=[]):
        """
        - Sends a sampled subset of the request batch to each shadow policy,
          and updates their hit-rate stats.
        - Applies the request batch to the active (production) cache.
        - Returns (cache_hits, evicted_items) from the active cache.
        """
        if self.active_policy is None:
            raise RuntimeError("SamplingMetaCache must be initialized before use.")

        embeds = np.asarray(embeds)
        embeds_ids = np.asarray(embeds_ids)

        batch_size = len(embeds)
        self.global_queries += batch_size

        # ---- 1) Sampling to shadow policies ----
        for idx, policy in enumerate(self.shadow_policies):
            # Decide which requests go to this shadow policy.
            if self.sample_prob >= 1.0:
                mask = np.ones(batch_size, dtype=bool)
            else:
                mask = np.random.rand(batch_size) < self.sample_prob

            if not mask.any():
                continue

            sampled_embeds = embeds[mask]
            sampled_ids = embeds_ids[mask]

            if texts:
                sampled_texts = [t for t, m in zip(texts, mask) if m]
            else:
                sampled_texts = []

            hits_shadow, _ = policy.cache(
                sampled_embeds,
                sampled_ids,
                count_nn=count_nn,
                texts=sampled_texts,
            )

            hits_sum = int(np.sum(hits_shadow))
            queries_count = len(sampled_embeds)
            self.policy_stats[idx]["hits"] += hits_sum
            self.policy_stats[idx]["queries"] += queries_count

        # ---- 2) Active (production) cache ----
        hits_prod, evicted_prod = self.active_policy.cache(
            embeds,
            embeds_ids,
            count_nn=count_nn,
            texts=texts,
        )

        self.total_hits += int(np.sum(hits_prod))
        self.total_queries += batch_size

        # ---- 3) Possibly switch the active policy ----
        self._maybe_switch_active_policy()

        return hits_prod, evicted_prod
