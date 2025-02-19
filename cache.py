import heapq
import random
from collections import OrderedDict, defaultdict, deque
import numpy as np
from scipy.spatial import distance_matrix
import faiss


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
        dists = np.sqrt(dists)
        return dists, ids

    def size(self):
        return len(self.items)


class Dummy(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        super().initialize(capacity, index)

    def request(self, embeds, embeds_ids, count_nn=1):
        shape = (len(embeds),)
        return np.zeros(shape, dtype=bool), embeds_ids


class RR(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        # Use a plain dict mapping embed_id -> usage count.
        self.items = []
        super().initialize(capacity, index)

    def request(self, embeds, embeds_ids, count_nn=1):
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


class LFU(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        # Use a dict mapping embed_id -> frequency (access count)
        self.items = {}
        super().initialize(capacity, index)


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
            cand = closest_ids[i_embed][i_nn]
            self.items[cand] += 1
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

class DistanceLFU(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        self.items = {}
        super().initialize(capacity, index)
    
    def request(self, embeds, embeds_ids, count_nn=1):
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
    
class LRU(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        self.items = OrderedDict()
        super().initialize(capacity, index)

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
            nn = closest_ids[i_embed][i_nn]
            self.items.move_to_end(nn)
        for embed_id, embed in zip(embeds_ids, embeds):
            self.items[embed_id] = None
            additions_embeds.append(embed)
            additions_ids.append(embed_id)
        if additions_ids:
            self.index.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        for _ in range(count_remove):
            embed_id, _ = self.items.popitem()
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

    def request(self, embeds, embeds_ids, count_nn=1):
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

    def request(self, embeds, embeds_ids, count_nn=1):
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

    def request(self, embeds, embeds_ids, count_nn=1):
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

class OPT(Cache):
    def __init__(self, same_embed_distance, embeds):
        super().__init__(same_embed_distance)
        self.embeds_covers = self.create_embeds_covers_faiss(embeds, same_embed_distance)

    def create_embeds_covers_faiss(self, embeds, same_embed_distance):
        embeds = np.asarray(embeds, dtype=np.float32)
        n, d = embeds.shape
        index = faiss.IndexFlatL2(d)
        index.add(embeds)
        threshold = same_embed_distance ** 2
        lims, distances, indices = index.range_search(embeds, threshold)
        embeds_covers = np.zeros((n, n), dtype=int)
        for i in range(n):
            start = lims[i]
            end = lims[i + 1]
            for pos in range(start, end):
                j = indices[pos]
                if j > i:
                    embeds_covers[i, j] = 1
        return embeds_covers

    def initialize(self, capacity: int, index):
        self.items = {}
        self.curr_embed_id = 0
        super().initialize(capacity, index)
    
    def get_next_hit(self, embed_id):
        row = self.embeds_covers[embed_id]
        i = self.curr_embed_id
        sub_row = row[i+1:]
        if np.any(sub_row):
            return np.argmax(sub_row) + (i + 1)
        else:
            return float('inf')
        
    def request(self, embeds, embeds_ids, count_nn=1):
        closest_dists, closest_ids = self.get_closest_stored_embeds(embeds, count_nn)
        cache_hits = np.sum(closest_dists < self.same_embed_distance, axis=1)
        evicted_items = []
        rejected_items = []
        additions = []
        for i_embed, (embed, embed_id) in enumerate(zip(embeds, embeds_ids)):
            self.curr_embed_id = embed_id
            embed_next_hit = self.get_next_hit(embed_id)
            self.items = {eid: self.get_next_hit(eid) if next_hit <= self.curr_embed_id else next_hit for eid, next_hit in self.items.items()}
            max_next_hit_embed_id = max(self.items, key=self.items.get, default=None)
            max_next_hit = self.items.get(max_next_hit_embed_id, float('inf'))
            if self.capacity > self.size() or (embed_next_hit < max_next_hit and embed_next_hit not in self.items.values()):
                if self.capacity <= self.size():
                    evicted_items.append(max_next_hit_embed_id)
                    self.items.pop(max_next_hit_embed_id, None)
                self.items[embed_id] = embed_next_hit
                additions.append((embed_id, embed))
            else: 
                rejected_items.append(embed_id)
        if additions:
            additions_embeds = [v for (_, v) in additions]
            additions_ids = [v for (v, _) in additions]
            self.index.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        if evicted_items:
            self.index.remove_ids(np.array(evicted_items))
        return cache_hits, evicted_items + rejected_items


class TinyLFU(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def decay_frequencies(self):
        #print("decay")
        evicted_items = []
        self.sample_counter *= self.decay_factor
        for key in list(self.freq_sketch.keys()):
            self.freq_sketch[key] = int(self.freq_sketch[key] * self.decay_factor)
            if self.freq_sketch[key] == 0:
                del self.freq_sketch[key]
                evicted_items.append(key)
        return evicted_items
    
    def initialize(self, capacity: int, index):
        self.decay_factor = 0.5
        self.sample_counter = 0
        self.sample_size = int(10 * capacity)
        self.freq_sketch = defaultdict(int)
        self.capacity = capacity
        self.index = index
        self.items = OrderedDict()

    def request(self, embeds, embeds_ids, count_nn=1):
        closest_dists, closest_ids = self.get_closest_stored_embeds(embeds, count_nn)
        mask = closest_dists < self.same_embed_distance
        cache_hits_indices = np.where(mask)
        cache_hits = np.count_nonzero(mask, axis=1)
        removals = []
        rejects = []
        additions_embeds = []
        additions_ids = []
        cache_hits_embeds_ids = [closest_ids[i][i_nn] for i, i_nn in zip(*cache_hits_indices)] + embeds_ids
        cache_hits_embeds = [None for _ in range(len(cache_hits_indices[0]))] + list(embeds)
        for embed_id, embed in zip(cache_hits_embeds_ids, cache_hits_embeds):
            self.sample_counter += 1
            if self.sample_counter >= self.sample_size:
                removals += self.decay_frequencies()
            self.freq_sketch[embed_id] += 1
            if embed_id in self.items:
                self.items.move_to_end(embed_id)
            elif self.size() <= self.capacity:
                self.items[embed_id] = None
                additions_ids.append(embed_id)
                additions_embeds.append(embed)
            else:
                victim_embed_id, _ = next(iter(self.items.items()))
                victim_freq = self.freq_sketch[victim_embed_id]
                embed_freq = self.freq_sketch[embed_id]
                if embed_freq >= victim_freq:
                    self.items.popitem(last=False)
                    removals.append(victim_embed_id)
                    self.items[embed_id] = None
                    additions_ids.append(embed_id)
                    additions_embeds.append(embed)
                else:
                    rejects.append(embed_id)
        if removals:
            #print("remove", removals)
            self.index.remove_ids(np.array(removals))
        if additions_ids:
            #print("add", additions_ids)
            self.index.add_with_ids(np.array(additions_embeds), np.array(additions_ids))

        return cache_hits, removals + rejects
