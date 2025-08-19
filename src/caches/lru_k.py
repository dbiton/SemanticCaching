from collections import defaultdict, deque
from caches.cache import Cache
import numpy as np

class LRUK(Cache):
    def __init__(self, same_embed_distance, K: int = 2):
        super().__init__(same_embed_distance)
        assert K >= 1
        self.K = K
        self.access_history = defaultdict(lambda: deque(maxlen=K))
        self.clock = 0

    def initialize(self, capacity: int, index):
        self.items = {}
        super().initialize(capacity, index)

    def _touch(self, embed_id):
        self.clock += 1
        self.access_history[embed_id].append(self.clock)

    def _score(self, embed_id):
        h = self.access_history.get(embed_id, [])
        if len(h) < self.K:
            return -1
        return h[0]

    def request(self, embeds, embeds_ids, count_nn=1, texts=[]):
        closest_dists, closest_ids = self.get_closest_stored_embeds(embeds, count_nn)
        hits_mask = closest_dists < self.same_embed_distance
        cache_hits = np.count_nonzero(hits_mask, axis=1)

        for i_row, i_nn in zip(*np.where(hits_mask)):
            self._touch(int(closest_ids[i_row, i_nn]))

        new_insertions = []
        for emb, eid in zip(embeds, embeds_ids):
            if eid in self.items:
                raise RuntimeError("embed id should be unique!")
            new_insertions.append((eid, emb))

        evicted_items = []
        n_evictions = max(0, self.size() + len(new_insertions) - self.capacity)
        if n_evictions:
            victims = sorted(self.items.keys(), key=self._score)[:n_evictions]
            self.remove_ids(np.array(victims, dtype=np.int64))
            for vid in victims:
                evicted_items.append(vid)
                self.items.pop(vid, None)
                self.access_history.pop(vid, None)

        if new_insertions:
            ins_embeds  = np.array([e for _, e in new_insertions])
            ins_ids     = np.array([i for i, _ in new_insertions])
            self.add_with_ids(ins_embeds, ins_ids)
            for eid in ins_ids:
                self.items[eid] = None
                self._touch(int(eid))

        return cache_hits, evicted_items
