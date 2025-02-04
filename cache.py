import random
from collections import OrderedDict
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

    def request(self, embeds, embeds_ids):
        raise NotImplementedError("virtual method")

    def get_closest_stored_embeds(self, embeds, count_nn=1):
        # Do a single batched search in the index.
        return self.index.search(embeds, count_nn)

    def size(self):
        return len(self.items)


class Dummy(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        super().initialize(capacity, index)

    def request(self, embeds, embeds_ids):
        return [False for _ in embeds_ids], embeds_ids


class RR(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)

    def initialize(self, capacity: int, index):
        # Use a plain dict mapping embed_id -> usage count.
        self.items = {}
        super().initialize(capacity, index)

    def request(self, embeds, embeds_ids):
        # One batched call to the index:
        closest_dists, closest_ids = self.get_closest_stored_embeds(embeds)
        cache_hits = []
        evicted_items = (
            []
        )  # one evicted id per request (if no eviction then simply return the new embed id)
        removals = []  # list of embed ids to remove from the index
        additions_embeds = []  # list of embeddings to add
        additions_ids = []  # list of corresponding embed ids

        for i, embed_id in enumerate(embeds_ids):
            hit = False
            # By default, if nothing is evicted we report the new id:
            evicted = embed_id
            cand = closest_ids[i][0]
            cand_dist = closest_dists[i][0]
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
            cache_hits.append(hit)
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

    def request(self, embeds, embeds_ids):
        closest_dists, closest_ids = self.get_closest_stored_embeds(embeds)
        cache_hits = []
        evicted_items = []
        removals = []
        additions_embeds = []
        additions_ids = []

        for i, embed_id in enumerate(embeds_ids):
            hit = False
            evicted = embed_id
            cand = closest_ids[i][0]
            cand_dist = closest_dists[i][0]
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
            cache_hits.append(hit)
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

    def request(self, embeds, embeds_ids):
        closest_dists, closest_ids = self.get_closest_stored_embeds(embeds)
        cache_hits = []
        evicted_items = []
        removals = []
        additions_embeds = []
        additions_ids = []

        for i, embed_id in enumerate(embeds_ids):
            hit = False
            evicted = embed_id
            cand = closest_ids[i][0]
            cand_dist = closest_dists[i][0]
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
            cache_hits.append(hit)
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

    def request(self, embeds, embeds_ids):
        closest_dists, closest_ids = self.get_closest_stored_embeds(embeds)
        cache_hits = []
        evicted_items = []
        removals = []
        additions_embeds = []
        additions_ids = []

        for i, embed_id in enumerate(embeds_ids):
            hit = False
            evicted = embed_id
            cand = closest_ids[i][0]
            cand_dist = closest_dists[i][0]
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
                    cache_hits.append(hit)
                    evicted_items.append(embed_id)
                    continue
                else:
                    del self.items[candidate]
                    removals.append(candidate)
                    evicted = candidate
            self.items[embed_id] = 1
            additions_embeds.append(embeds[i])
            additions_ids.append(embed_id)
            cache_hits.append(hit)
            evicted_items.append(evicted)

        if removals:
            self.index.remove_ids(np.array(removals))
        if additions_ids:
            self.index.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        return cache_hits, evicted_items
