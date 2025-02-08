import heapq
import random
from collections import OrderedDict, defaultdict
import numpy as np
from scipy.spatial import distance_matrix


class Cache:
    def __init__(self, same_embed_distance: float):
        self.items = None  # our internal cache storage
        self.capacity = None
        self.index = None  # an external “index” object
        self.same_embed_distance = same_embed_distance

    def initialize(self, capacity: int, index):
        self.capacity = capacity
        self.index = index

    def request(self, embeds, embeds_ids, count_nn=1):
        raise NotImplementedError("virtual method")

    def get_closest_stored_embeds(self, embeds, count_nn=1):
        dists, ids = self.index.search(embeds, count_nn)
        return dists, ids

    def size(self):
        return len(self.items)


class Dummy(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        super().initialize(capacity, index)

    def request(self, embeds, embeds_ids, count_nn=1):
        shape = (len(embeds), count_nn)
        return np.zeros(shape, dtype=bool), embeds_ids


class RR(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        # Use a plain dict mapping embed_id -> usage count.
        self.items = {}
        super().initialize(capacity, index)

    def request(self, embeds, embeds_ids, count_nn=1):
        # One batched call to the index:
        closest_dists, closest_ids = self.get_closest_stored_embeds(embeds, count_nn)
        cache_hits = np.zeros((len(embeds), count_nn), dtype=bool)
        evicted_items = (
            []
        )  # one evicted id per request (if no eviction then simply return the new embed id)
        removals = []  # list of embed ids to remove from the index
        additions_embeds = []  # list of embeddings to add
        additions_ids = []  # list of corresponding embed ids

        for i, embed_id in enumerate(embeds_ids):
            for i_nn in range(len(closest_ids[i])):
                hit = False
                # By default, if nothing is evicted we report the new id:
                evicted = embed_id
                cand = closest_ids[i][i_nn]
                cand_dist = closest_dists[i][i_nn]
                if cand in self.items and cand_dist < self.same_embed_distance:
                    self.items[cand] += 1
                    hit = True
                # If our cache is full, remove one random element:
                if self.size() >= self.capacity:
                    rem = random.choice(list(self.items.keys()))
                    del self.items[rem]
                    removals.append(rem)
                    evicted = rem
                # Always add the new item:
                self.items[embed_id] = 1
                additions_embeds.append(embeds[i])
                additions_ids.append(embed_id)
                cache_hits[i][i_nn] = hit
                evicted_items.append(evicted)

        if removals:
            self.index.remove_ids(np.array(removals))
        if additions_ids:
            self.index.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        return cache_hits, evicted_items


class LFU(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        # Use a dict mapping embed_id -> frequency (access count)
        self.items = {}
        super().initialize(capacity, index)

    def request(self, embeds, embeds_ids, count_nn=1):
        closest_dists, closest_ids = self.get_closest_stored_embeds(embeds, count_nn)
        cache_hits = np.zeros((len(embeds), count_nn), dtype=bool)
        evicted_items = []
        removals = []
        additions_embeds = []
        additions_ids = []

        for i, embed_id in enumerate(embeds_ids):
            for i_nn in range(len(closest_ids[i])):
                hit = False
                evicted = embed_id
                cand = closest_ids[i][i_nn]
                cand_dist = closest_dists[i][i_nn]
                if cand in self.items and cand_dist < self.same_embed_distance:
                    self.items[cand] += 1
                    hit = True
                # Evict the least–frequently used item if the cache is full.
                if self.size() >= self.capacity:
                    rem = min(self.items, key=self.items.get)
                    del self.items[rem]
                    removals.append(rem)
                    evicted = rem
                self.items[embed_id] = 1
                additions_embeds.append(embeds[i])
                additions_ids.append(embed_id)
                cache_hits[i][i_nn] = hit
                evicted_items.append(evicted)

        if removals:
            self.index.remove_ids(np.array(removals))
        if additions_ids:
            self.index.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        return cache_hits, evicted_items


class LRU(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        # Use an OrderedDict to keep the items sorted by most recent use.
        self.items = OrderedDict()
        super().initialize(capacity, index)

    def request(self, embeds, embeds_ids, count_nn=1):
        closest_dists, closest_ids = self.get_closest_stored_embeds(embeds, count_nn)
        cache_hits = np.zeros((len(embeds), count_nn), dtype=bool)
        evicted_items = []
        removals = []
        additions_embeds = []
        additions_ids = []

        for i, embed_id in enumerate(embeds_ids):
            for i_nn in range(len(closest_ids[i])):
                hit = False
                evicted = embed_id
                cand = closest_ids[i][i_nn]
                cand_dist = closest_dists[i][i_nn]
                if cand in self.items and cand_dist < self.same_embed_distance:
                    # Move the item to the end to mark it as recently used.
                    self.items.move_to_end(cand)
                    hit = True
                if self.size() >= self.capacity:
                    # Remove the least–recently used item.
                    rem, _ = self.items.popitem(last=False)
                    removals.append(rem)
                    evicted = rem
                self.items[embed_id] = None  # value not used
                additions_embeds.append(embeds[i])
                additions_ids.append(embed_id)
                cache_hits[i][i_nn] = hit
                evicted_items.append(evicted)

        if removals:
            self.index.remove_ids(np.array(removals))
        if additions_ids:
            self.index.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        return cache_hits, evicted_items

class RAP(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        # Again, we use a dict mapping embed_id -> frequency.
        self.items = {}
        super().initialize(capacity, index)

    def request(self, embeds, embeds_ids, count_nn=1):
        closest_dists, closest_ids = self.get_closest_stored_embeds(embeds, count_nn)
        cache_hits = np.zeros((len(embeds), count_nn), dtype=bool)
        evicted_items = []
        removals = []
        additions_embeds = []
        additions_ids = []

        for i, embed_id in enumerate(embeds_ids):
            for i_nn in range(len(closest_ids[i])):
                hit = False
                evicted = embed_id
                cand = closest_ids[i][i_nn]
                cand_dist = closest_dists[i][i_nn]
                if cand in self.items and cand_dist < self.same_embed_distance:
                    self.items[cand] += 1
                    hit = True
                if self.size() >= self.capacity:
                    # Choose the least–used candidate.
                    candidate = min(self.items, key=self.items.get)
                    cand_hits = self.items[candidate]
                    thresh = 1.0 / (cand_hits + 1)
                    # With probability (1 - thresh) do not cache the new embed.
                    if random.random() >= thresh:
                        cache_hits[i][i_nn] = hit
                        evicted_items.append(embed_id)
                        continue
                    else:
                        del self.items[candidate]
                        removals.append(candidate)
                        evicted = candidate
                self.items[embed_id] = 1
                additions_embeds.append(embeds[i])
                additions_ids.append(embed_id)
                cache_hits[i][i_nn] = hit
                evicted_items.append(evicted)

        if removals:
            self.index.remove_ids(np.array(removals))
        if additions_ids:
            self.index.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        return cache_hits, evicted_items

class FixedRadius(Cache):
    def __init__(self, same_embed_distance, radius):
        super().__init__(same_embed_distance)
        self.radius = radius

    def initialize(self, capacity: int, index):
        self.items = {}
        super().initialize(capacity, index)

    def request(self, embeds, embeds_ids, count_nn=1):
        closest_dists, closest_ids = self.get_closest_stored_embeds(embeds, count_nn)
        cache_hits = np.zeros((len(embeds), count_nn), dtype=bool)
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
                if nn_embed_distance <= self.same_embed_distance:
                    cache_hits[i][i_nn] = True
            if np.sum(cache_hits[i]) == 0 or self.size() < self.capacity:
                additions.append((embed_id, embeds[i], size_neigh))
            else:
                evicted_items.append(embed_id)
        count_removed = (len(additions) + self.size()) - self.capacity        
        if count_removed > 0:
            removed_items = heapq.nsmallest(count_removed, self.items, key=self.items.get)
            for k in removed_items:
                del self.items[k]
            self.index.remove_ids(np.array(removed_items))
            evicted_items += removed_items
        if additions:
            additions_embeds = [v for (_, v, _) in additions]
            additions_ids = [v for (v, _, _) in additions]
            for (embed_id, _, size_neigh) in additions:
                self.items[embed_id] = size_neigh
            self.index.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        return cache_hits, evicted_items

import faiss
class PCA(Cache):
    def __init__(self, same_embed_distance, pca_dim = 9, cluster_diameter = 1.0):
        super().__init__(same_embed_distance)
        self.pca_dim = pca_dim
        self.train_counter = 0
        self.cluster_diameter = cluster_diameter

    def train_pca(self):
        n_embeds = self.index.ntotal
        dim_embeds = self.index.d
        faiss.PCAMatrix(dim_embeds, self.pca_dim)
        embeds = np.zeros((n_embeds, dim_embeds), dtype='float32')
        faiss.vector_to_array(self.index.reconstruct_n(0, n_embeds), embeds)
        self.pca.train(embeds)
        assert self.pca.is_trained
        embeds_pca = self.pca.apply(embeds)
        embeds_ids = self.index.id_map[:]
        self.clusters = defaultdict(list)
        for embed_id, embed_pca in zip(embeds_ids, embeds_pca):
            cluster_id = self.embed_pca_to_cluster_id(embed_pca)
            self.clusters[cluster_id].append(embed_id)
            
    def embed_pca_to_cluster_id(self, embed_pca):
        unit_hypercube_diameter = np.sqrt(self.pca_dim)
        hypercube_side_length = self.cluster_diameter / unit_hypercube_diameter
        np.round(embed_pca / hypercube_side_length).astype(int)
    
    def initialize(self, capacity: int, index):
        self.items = {}
        super().initialize(capacity, index)

    def request(self, embeds, embeds_ids, count_nn=1):
        closest_dists, closest_ids = self.get_closest_stored_embeds(embeds, count_nn)
        cache_hits = np.zeros((len(embeds), count_nn), dtype=bool)
        evicted_items = []
        additions = []
        for i, embed_id in enumerate(embeds_ids):
            embed_closest_ids = closest_ids[i]
            embed_closest_distances = closest_dists[i]
            size_neigh = np.sum(embed_closest_distances < self.radius)
            for i_nn, (nn_embed_id, nn_embed_distance) in enumerate(zip(embed_closest_ids, embed_closest_distances)):
                if nn_embed_distance <= self.same_embed_distance:
                    cache_hits[i][i_nn] = True
            additions.append((embed_id, embeds[i], size_neigh))
        count_removed = (len(additions) + self.size()) - self.capacity        
        if count_removed > 0:
            for _ in range(count_removed):
                smallest_cluster = min(self.clusters, key=lambda k: len(self.clusters[k]))
                k = self.clusters[smallest_cluster].pop()
                del self.items[k]
                evicted_items.append(k)
            self.index.remove_ids(np.array(evicted_items))
        if additions:
            additions_embeds = [v for (_, v, _) in additions]
            additions_ids = [v for (v, _, _) in additions]
            for (embed_id, _, size_neigh) in additions:
                self.items[embed_id] = size_neigh
            self.index.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        return cache_hits, evicted_items