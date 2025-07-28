from collections import defaultdict
from heapq import heapify, heappop, heappush
import random
from typing import Tuple
import numpy as np
import faiss
from cache import Cache

# First fill up the cache, afterwards cluster and choose how to split stuff
class CacheKMeans(Cache):
    def __init__(
        self,
        same_embed_distance: float,
        ratio_clusters: float,
        count_iter: int
    ):
        super().__init__(same_embed_distance)
        self.kmeans = None
        # Maps cluster_id -> set of embed_ids
        self.ratio_clusters = ratio_clusters
        self.count_iter = count_iter
        self.kmeans = None
        self.clusters = None
        self.items = None

    def initialize(self, capacity: int, index):
        super().initialize(capacity, index)
        dim_embeds = self.index.d
        count_clusters = capacity // 40
        self.kmeans = faiss.Kmeans(dim_embeds, count_clusters, niter=self.count_iter, verbose=False)
        self.clusters = defaultdict(set)
        self.items = {}
        self.marked = {}

    def train_kmeans(self):
        n_embeds = self.index.ntotal
        embeds = self.index.index.reconstruct_n(0, n_embeds)
        embeds_ids = faiss.vector_to_array(self.index.id_map)
        self.kmeans.train(embeds)
        distances, embeds_centroids = self.kmeans.index.search(embeds, 1)
        clusters = defaultdict(list)
        for embed_id, embed_centroid, embed_distance in zip(embeds_ids, embeds_centroids, distances):
            clusters[tuple(embed_centroid)].append(embed_id)
        N = int(0.2 * n_embeds)
        heap = [(len(lst), key) for key, lst in clusters.items() if lst]
        heapify(heap)
        self.marked = []
        while N > 0 and heap:
            _, key = heappop(heap)
            if clusters[key]:  # Ensure the list is not empty
                self.marked.append(clusters[key].pop(0))  # Take from the front
                N -= 1
                if clusters[key]:
                    heappush(heap, (len(clusters[key]), key))
            
    def request(self, embeds, embeds_ids, count_nn=1):
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
        count_remove = (len(additions) + self.size()) - self.capacity        
        while count_remove > 0:
            if len(self.marked) == 0:
                self.train_kmeans()
            removed = self.marked[:count_remove]
            self.marked = self.marked[count_remove:]
            count_remove -= len(removed)
            evicted_items += removed
        if len(evicted_items) > 0:
            self.index.remove_ids(np.array(evicted_items))
            for v in evicted_items:
                if v in self.items:
                    self.items.pop(v)
        if additions:
            additions_embeds = [v for (_, v) in additions]
            additions_ids = [v for (v, _) in additions]
            for (embed_id, embed) in additions:
                self.items[embed_id] = embed
            self.index.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        return cache_hits, evicted_items