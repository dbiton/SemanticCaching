import heapq
import random
from collections import OrderedDict, defaultdict, deque
import heapq
import numpy as np
from scipy.spatial import distance_matrix
import faiss

from util.surprisal import calculate_perplexity, calculate_surprisal
from util.online_clusters import OnlineClusters


class Cache:
    def __init__(self, same_embed_distance: float):
        self.items = None  # our internal cache storage
        self.capacity = None
        self.index = None  # an external “index” object
        self.same_embed_distance = same_embed_distance

    def initialize(self, capacity: int, index):
        self.capacity = capacity
        self.index = index

    def request(self, embeds, embeds_ids, count_nn=1, texts=[]):
        raise NotImplementedError("virtual method")

    def get_closest_stored_embeds(self, embeds, count_nn=1):
        dists, ids = self.index.search(embeds, count_nn)
        dists = np.sqrt(dists)
        return dists, ids
    
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

    def request(self, embeds, embeds_ids, count_nn=1, texts=[]):
        shape = (len(embeds),)
        return np.zeros(shape, dtype=bool), embeds_ids


class RR(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        # Use a plain dict mapping embed_id -> usage count.
        self.items = []
        super().initialize(capacity, index)

    def request(self, embeds, embeds_ids, count_nn=1, texts=[]):
        assert(len(embeds) < self.capacity)
        closest_dists, _ = self.get_closest_stored_embeds(embeds, count_nn)
        cache_hits = np.sum(closest_dists < self.same_embed_distance, axis=1)
        count_remove = max(0, (len(embeds) + self.size()) - self.capacity)
        removed_indices = sorted(random.sample(range(len(self.items)), count_remove), reverse=True)
        removals = np.array([self.items[i] for i in removed_indices])
        for i in removed_indices:
            self.items.pop(i)
        if len(removals) > 0:
            self.index.remove_ids(removals)
        self.index.add_with_ids(embeds, embeds_ids)
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

    def request(self, embeds, embeds_ids, count_nn=1, texts=[]):
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
            self.index.remove_ids(np.array(evicted_items))
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

    def request(self, embeds, embeds_ids, count_nn=1, texts=[]):
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
        if evicted_items:
            self.index.remove_ids(np.array(evicted_items))
        if additions_ids:
            self.index.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        return cache_hits, evicted_items

class SimpleCache(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def get_score(self, embed, embed_id: int, embed_text: str) -> float:
        return 0
    
    def initialize(self, capacity: int, index):
        self.items = {}
        super().initialize(capacity, index)

    def request(self, embeds, embeds_ids, count_nn=1, texts=[]):
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
            self.index.remove_ids(np.array(evicted_items))
        if additions_ids:
            self.index.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
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
    
    def request(self, embeds, embeds_ids, count_nn=1, texts=[]):
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
            self.index.remove_ids(np.array(evicted_items))
        if additions_ids:
            self.index.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        return cache_hits, evicted_items

class DistanceLFU(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        self.items = {}
        super().initialize(capacity, index)
    
    def request(self, embeds, embeds_ids, count_nn=1, texts=[]):
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
            self.index.remove_ids(np.array(evicted_items))
        if additions_ids:
            self.index.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        return cache_hits, evicted_items

class DynamicAgingLFU(Cache):
    def __init__(self, same_embed_distance, half_life: int):
        super().__init__(same_embed_distance)
        self.decay = pow(0.5, 1 / half_life)

    def initialize(self, capacity: int, index):
        self.items = {}
        super().initialize(capacity, index)
    
    def decay_counters(self):
        for key in self.items:
            self.items[key] *= self.decay
    
    def request(self, embeds, embeds_ids, count_nn=1, texts=[]):
        self.decay_counters()        
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
            self.items[embed_id] = 1
            additions_embeds.append(embed)
            additions_ids.append(embed_id)
        if evicted_items:
            self.index.remove_ids(np.array(evicted_items))
        if additions_ids:
            self.index.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        return cache_hits, evicted_items

class LRU(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        self.items = OrderedDict()
        super().initialize(capacity, index)

    def request(self, embeds, embeds_ids, count_nn=1, texts=[]):
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
            self.index.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        for _ in range(count_remove):
            embed_id, _ = self.items.popitem(last = False)
            evicted_items.append(embed_id)
        if evicted_items:
            self.index.remove_ids(np.array(evicted_items))
        return cache_hits, evicted_items

class RAP(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        self.items = {}
        super().initialize(capacity, index)

    def request(self, embeds, embeds_ids, count_nn=1, texts=[]):
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
            self.index.remove_ids(np.array(removed_items))
        if additions_ids:
            self.items.update({eid: 0 for eid in additions_ids})
            self.index.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        return cache_hits, removed_items + rejected_items

class FixedRadius(Cache):
    def __init__(self, same_embed_distance, radius):
        super().__init__(same_embed_distance)
        self.radius = radius

    def initialize(self, capacity: int, index):
        self.items = {}
        super().initialize(capacity, index)

    def request(self, embeds, embeds_ids, count_nn=1, texts=[]):
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
            self.index.remove_ids(np.array(removed_items))
            evicted_items += removed_items
        if additions:
            additions_embeds = [v for (_, v, _) in additions]
            additions_ids = [v for (v, _, _) in additions]
            for (embed_id, _, size_neigh) in additions:
                self.items[embed_id] = size_neigh
            self.index.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
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

    def request(self, embeds, embeds_ids, count_nn=1, texts=[]):
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
            self.index.remove_ids(np.array(evicted_items))
        if additions:
            additions_embeds = [v for (_, v) in additions]
            additions_ids = [v for (v, _) in additions]
            additions_clusters_ids = self.embeds_to_clusters_ids(additions_embeds)
            for (embed_id, embed), cluster_id in zip(additions, additions_clusters_ids):
                self.items[embed_id] = cluster_id
            self.index.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
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
    
    def request(self, embeds, embeds_ids, count_nn=1, texts=[]):
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
            self.index.remove_ids(np.array(removals))
        if additions_ids:
            self.index.add_with_ids(np.array(additions_embeds), np.array(additions_ids))

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

    def request(self, embeds, embeds_ids, count_nn=1, texts=[]):
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
            self.index.remove_ids(np.array(removals))
        if additions_ids:
            self.index.add_with_ids(np.array(additions_embeds), np.array(additions_ids))

        return cache_hits, removals