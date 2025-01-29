from collections import defaultdict
import random
from typing import Tuple
import numpy as np
from river import cluster
from cache import Cache

class DenStreamCache(Cache):
    def __init__(
        self,
        same_embed_distance: float,
    ):
        super().__init__(same_embed_distance)
        self.denstream = None
        # Maps cluster_id -> set of embed_ids
        self.cluster_members = defaultdict(set)

    def initialize(self, capacity: int, index):
        super().initialize(capacity, index)
        self.denstream = cluster.DenStream(
            mu=8,
            epsilon = 0.75,
            beta=0.5,
            n_samples_init=capacity//2
        )
        self.cluster_members.clear()

    def size(self):
        return sum(len(members) for members in self.cluster_members.values())

    def request(self, embed: np.ndarray, embed_id: int) -> Tuple[bool, int]:
        closest_embed_id, closest_embed_distance = self.get_closest_stored_embed(embed)
        cache_hit = False
        evicted_item = None
        if closest_embed_id is not None and closest_embed_distance < self.same_embed_distance:
            cache_hit = True
        if not cache_hit and self.size() >= self.capacity:
            evicted_item = self.evict_from_least_dense_cluster()
        x = {f"dim_{i}": float(v) for i, v in enumerate(embed.flatten())}
        self.denstream.learn_one(x)
        cluster_id = self.denstream.predict_one(x)
        self.cluster_members[cluster_id].add(embed_id)
        self.index.add_with_ids(embed, np.array([embed_id]))
        return cache_hit, evicted_item

    def evict_from_least_dense_cluster(self) -> int:
        if -1 in self.cluster_members:
            selected_cluster_id = -1
            selected_cluster = self.cluster_members[-1]
        else:
            selected_cluster_id = min(self.cluster_members, key=lambda k: len(self.cluster_members[k]))
            selected_cluster = self.cluster_members[selected_cluster_id]
        evicted_embed_id = random.choice(list(selected_cluster))
        selected_cluster.remove(evicted_embed_id)
        if len(selected_cluster) == 0:
            self.cluster_members.pop(selected_cluster_id)
        self.index.remove_ids(np.array([evicted_embed_id]))
        return evicted_embed_id

    def get_closest_stored_embed(self, embed: np.ndarray):
        if self.size() == 0:
            return None, float('inf')
        distances, neighbors = self.index.search(embed, 1)
        return neighbors[0][0], distances[0][0]
