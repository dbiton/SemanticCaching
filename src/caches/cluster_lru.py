from collections import OrderedDict
from caches.cache import Cache
import numpy as np
from src.util.online_clusters import OnlineClusters

# ---------------------------------------------------------------------
# Cluster‑aware LRU cache
# ---------------------------------------------------------------------
class ClusterLRU(Cache):
    """
    Evicts vectors from the least‑recently‑used *cluster*.
    Recency is tracked per‑cluster; a hit to any vector of a cluster
    makes the whole cluster most‑recent.
    """
    def __init__(self, same_embed_distance, embed_dim: int = 384):
        super().__init__(same_embed_distance)
        self.online_clusters = OnlineClusters(same_embed_distance, embed_dim)

    # -----------------------------------------------------------------
    def initialize(self, capacity: int, index):
        self.items            = set()        # just embed IDs for size checks
        self.cluster_recency  = OrderedDict()# cid → None (LRU order)
        super().initialize(capacity, index)

    # -----------------------------------------------------------------
    def _touch_cluster(self, cid):
        """Put cluster at MRU position."""
        if cid in self.cluster_recency:
            self.cluster_recency.move_to_end(cid)
        else:
            self.cluster_recency[cid] = None

    # -----------------------------------------------------------------
    def request(self, embeds, embeds_ids, count_nn=1, texts=[]):
        # ---------- 1. detect hits ------------------------------------
        closest_dists, closest_ids = self.get_closest_stored_embeds(embeds, count_nn)
        mask        = closest_dists < self.same_embed_distance
        cache_hits  = np.count_nonzero(mask, axis=1)

        hit_rows, hit_cols = np.where(mask)
        for r, c in zip(hit_rows, hit_cols):
            hit_id  = int(closest_ids[r, c])
            cid     = self.online_clusters.get_cluster(hit_id)
            self._touch_cluster(cid)

        # ---------- 2. plan evictions ---------------------------------
        need_space   = len(embeds)
        to_evict_cnt = max(0, len(self.items) + need_space - self.capacity)
        evicted_items = []

        while to_evict_cnt > 0 and self.cluster_recency:
            lru_cid, _      = self.cluster_recency.popitem(last=False)
            victim_id, empty = self.online_clusters.pop_vector(lru_cid)  # remove one vector
            evicted_items.append(victim_id)
            self.items.discard(victim_id)

            # cluster still has vectors → keep it in list as MRU
            if not empty:
                self._touch_cluster(lru_cid)

            to_evict_cnt -= 1

        if evicted_items:
            self.index.remove_ids(np.array(evicted_items, dtype=np.int64))

        # ---------- 3. insert new vectors -----------------------------
        additions_embeds = []
        additions_ids    = []

        for vec, eid in zip(embeds, embeds_ids):
            if eid in self.items:               # safeguard: no duplicates
                continue
            cid = self.online_clusters.add_vector(vec, eid)
            self.items.add(eid)
            self._touch_cluster(cid)

            additions_embeds.append(vec)
            additions_ids.append(eid)

        if additions_ids:
            self.index.add_with_ids(np.array(additions_embeds),
                                    np.array(additions_ids, dtype=np.int64))

        # ---------- 4. return ----------------------------------------
        return cache_hits, evicted_items
