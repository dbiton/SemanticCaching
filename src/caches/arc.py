from collections import OrderedDict
from caches.cache import Cache
import numpy as np

class ARC(Cache):
    def __init__(self, same_embed_distance):
        super().__init__(same_embed_distance)
        self.p = 0
        self.T1 = OrderedDict()          # recency list (id → vec)
        self.T2 = OrderedDict()          # frequency list (id → vec)
        self.B1 = OrderedDict()          # ghost for T1   (id → None)
        self.B2 = OrderedDict()          # ghost for T2   (id → None)

    # ------------------------------------------------------------------ #
    # helpers
    def _add_to_T1(self, eid, vec):
        self.T1[eid] = vec
        self.index.add_with_ids(vec[np.newaxis, :], np.array([eid]))

    def _add_to_T2(self, eid, vec):
        self.T2[eid] = vec
        self.index.add_with_ids(vec[np.newaxis, :], np.array([eid]))

    def _move_T1_to_T2(self, eid):
        vec = self.T1.pop(eid)
        self.T2[eid] = vec                    # already in FAISS

    def _replace(self, incoming_id):
        """
        Evict one real vector from T1 or T2 and move its ID to B1/B2.
        Returns the evicted ID (or None if nothing was evicted).
        """
        if self.T1 and (len(self.T1) > self.p or (incoming_id in self.B2 and len(self.T1) == self.p)):
            vid, vvec = self.T1.popitem(last=False)
            self.B1[vid] = None
        else:
            vid, vvec = self.T2.popitem(last=False)
            self.B2[vid] = None

        self.index.remove_ids(np.array([vid], dtype=np.int64))
        return vid
    # ------------------------------------------------------------------ #

    def initialize(self, capacity: int, index):
        self.capacity = capacity
        self.index    = index

    # ------------------------------------------------------------------ #
    def request(self, embeds, embeds_ids, count_nn=1, texts=[]):
        closest_dists, _ = self.get_closest_stored_embeds(embeds, 1)
        cache_hits = (closest_dists[:, 0] < self.same_embed_distance).astype(int)

        evicted_items = []   # <─ we will return this

        for vec, eid, hit in zip(embeds, embeds_ids, cache_hits):
            # ---------- hits -------------------------------------------
            if eid in self.T1:                           # Case I
                self._move_T1_to_T2(eid)
                continue
            if eid in self.T2:                           # Case II
                self.T2.move_to_end(eid)                 # MRU
                continue

            # ---------- history hits -----------------------------------
            if eid in self.B1:                           # Case III
                self.p = min(self.capacity, self.p + max(1, len(self.B2) // max(1, len(self.B1))))
                vid = self._replace(eid)
                if vid is not None:
                    evicted_items.append(vid)
                self.B1.pop(eid, None)
                self._add_to_T2(eid, vec)
                continue

            if eid in self.B2:                           # Case IV
                self.p = max(0, self.p - max(1, len(self.B1) // max(1, len(self.B2))))
                vid = self._replace(eid)
                if vid is not None:
                    evicted_items.append(vid)
                self.B2.pop(eid, None)
                self._add_to_T2(eid, vec)
                continue

            # ---------- brand‑new item -------------------------------
            if len(self.T1) + len(self.T2) >= self.capacity:
                if len(self.T1) < self.capacity:
                    # trim a ghost to make room, then replace a real vector
                    if len(self.T1) + len(self.B1) >= self.capacity:
                        self.B1.popitem(last=False)
                    else:
                        self.B2.popitem(last=False)
                    vid = self._replace(eid)
                    if vid is not None:
                        evicted_items.append(vid)
                else:
                    # T1 is full → move its LRU to B1
                    vid, vvec = self.T1.popitem(last=False)
                    self.B1[vid] = None
                    self.index.remove_ids(np.array([vid], dtype=np.int64))
                    evicted_items.append(vid)

            elif len(self.T1) + len(self.T2) + len(self.B1) + len(self.B2) >= 2 * self.capacity:
                # metadata overflow: drop oldest from B2
                self.B2.popitem(last=False)

            # finally insert the new vector into T1
            self._add_to_T1(eid, vec)

        return cache_hits, evicted_items
