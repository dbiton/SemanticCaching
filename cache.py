import heapq
import random
from collections import OrderedDict, defaultdict
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
        cache_hits = np.zeros((len(embeds),), dtype=bool)
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
                if cand in self.items and cand_dist <= self.same_embed_distance:
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
                cache_hits[i] += hit
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
                if cand in self.items and cand_dist <= self.same_embed_distance:
                    #print("LFU hit", cand, embed_id)
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
            #print("LFU evict", removals)
        if additions_ids:
            self.index.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
            #print("LFU add", additions_ids)
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
                if cand in self.items and cand_dist <= self.same_embed_distance:
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
                if cand in self.items and cand_dist <= self.same_embed_distance:
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
        cache_hits = np.zeros((len(embeds), count_nn), dtype=bool)
        evicted_items = []
        additions = []
        for i, embed_id in enumerate(embeds_ids):
            embed_closest_ids = closest_ids[i]
            embed_closest_distances = closest_dists[i]
            for i_nn, (nn_embed_id, nn_embed_distance) in enumerate(zip(embed_closest_ids, embed_closest_distances)):
                if nn_embed_distance <= self.same_embed_distance:
                    cache_hits[i][i_nn] = True
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
        self.embeds_distances = distance_matrix(embeds, embeds)
        self.embeds_covers = (self.embeds_distances <= same_embed_distance).astype(int)
        tri_l = np.tril_indices_from(self.embeds_covers)
        self.embeds_covers[tri_l] = 0

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
        cache_hits = np.zeros((len(embeds), count_nn), dtype=bool)
        evicted_items = []
        rejected_items = []
        additions = []
        i_embed = -1
        for embed, embed_id in zip(embeds, embeds_ids):
            i_embed += 1
            for i_nn, nn_distance in enumerate(closest_dists[i_embed]):
                if nn_distance <= self.same_embed_distance:
                    #print("OPT hit", closest_ids[i_embed][i_nn], embed_id)
                    cache_hits[i_embed][i_nn] = True
            self.curr_embed_id = embed_id
            embed_next_hit = self.get_next_hit(embed_id)
            items = {eid: self.get_next_hit(eid) for eid in self.items.keys()}
            max_next_hit_embed_id = max(items, key=items.get, default=None)
            max_next_hit = items.get(max_next_hit_embed_id, float('inf'))
            if self.capacity > self.size() or (embed_next_hit < max_next_hit and embed_next_hit not in items.values()):
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
            #print("OPT add", additions_ids)
        if evicted_items:
            self.index.remove_ids(np.array(evicted_items))
            #print("OPT evict", evicted_items)
        return cache_hits, evicted_items + rejected_items