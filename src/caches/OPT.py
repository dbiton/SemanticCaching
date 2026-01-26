import math
import sys
sys.path.append("...")
sys.path.append("..")
sys.path.append(".")


import bisect
from collections import OrderedDict
import pandas as pd
from caches.cache import Cache
import numpy as np
import faiss
import lightgbm as lgb
import xgboost as xgb
import random
from scipy import stats

import re
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from src.util.reduce_dim import cluster_by_coloring_dsat_faiss_heap, cluster_complete_linkage_faiss
from vector_stores.naive_interface import NaiveVectorStore

import numpy as np
import faiss

class OPT(Cache):

    def __init__(self, same_embed_distance, embeds):
        super().__init__(same_embed_distance)
        self.embeds = np.asarray(embeds, dtype=np.float32)
        self.max_embed_id = len(embeds)
        self.embeds_covers = self.create_embeds_covers(self.embeds, same_embed_distance)
        self.curr_cover_counters = np.zeros(self.max_embed_id + 1, dtype=np.int32)

    def create_embeds_covers(self, embeds, same_embed_distance):
        n, d = embeds.shape
        index = faiss.IndexFlatL2(d)
        index.add(embeds)
        threshold = same_embed_distance ** 2
        lims, _, indices = index.range_search(embeds, threshold)
        
        embeds_covers = np.empty(n, dtype=object)
        for i in range(n):
            i_covers = np.sort(indices[lims[i]:lims[i + 1]])
            i_covers = i_covers[i_covers > i]
            embeds_covers[i] = i_covers
        return embeds_covers

    def initialize(self, capacity: int, index):
        self.items = {}
        self.curr_embed_id = 0
        self.curr_cover_counters.fill(0)
        super().initialize(capacity, index)
    
    def get_next_hits(self, embeds_ids):
        curr_id = self.curr_embed_id
        next_hits = {}
        batch_rows = self.embeds_covers[embeds_ids]        
        for embed_id, row in zip(embeds_ids, batch_rows):
            i = row.searchsorted(curr_id, side='right')
            next_hit = np.inf
            if len(row) > 0 and len(row) > i:
                next_hit = row[i]
            next_hits[embed_id] = next_hit
        return next_hits
    
    def cache(self, embeds, embeds_ids, count_nn=1, texts=[]):
        closest_dists, _ = self.get_closest_stored_embeds(embeds, count_nn)
        cache_hits = np.sum(closest_dists < self.same_embed_distance, axis=1)
        
        evicted_items = []
        rejected_items = []
        additions_embeds = []
        additions_ids = []
        
        self.curr_embed_id = max(embeds_ids)
        
        stale_items = [eid for (eid, next_hit) in self.items.items() if next_hit <= self.curr_embed_id]
        if stale_items:
            self.items.update(self.get_next_hits(stale_items))
        
        embeds_next_hits = self.get_next_hits(embeds_ids)
        
        for embed, embed_id in zip(embeds, embeds_ids):
            embed_next_hit = embeds_next_hits[embed_id] 
            if math.isinf(embed_next_hit):
                rejected_items.append(embed_id)
                continue
            max_next_hit_embed_id = max(self.items, key=self.items.get, default=None)
            max_next_hit = self.items.get(max_next_hit_embed_id, float('inf'))
            is_covered = self.curr_cover_counters[embed_next_hit] > 0
            if self.capacity > self.size() or (embed_next_hit < max_next_hit and not is_covered):
                if self.capacity <= self.size():
                    evicted_items.append(max_next_hit_embed_id)
                    self.items.pop(max_next_hit_embed_id)
                    evicted_covers = self.embeds_covers[max_next_hit_embed_id]
                    self.curr_cover_counters[evicted_covers] -= 1
                self.items[embed_id] = embed_next_hit
                new_covers = self.embeds_covers[embed_id]
                self.curr_cover_counters[new_covers] += 1
                additions_embeds.append(embed)
                additions_ids.append(embed_id)
            else: 
                rejected_items.append(embed_id)
        if additions_ids:
            self.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        if evicted_items:
            self.remove_ids(np.array(evicted_items))
            
        return cache_hits, evicted_items + rejected_items
class FGRVB(Cache):

    def __init__(self, same_embed_distance, embeds):
        super().__init__(same_embed_distance)
        # Pre-convert to float32 once to avoid repeated casting
        self.embeds = np.asarray(embeds, dtype=np.float32)
        self.embeds_covers = self.create_embeds_covers(self.embeds, same_embed_distance)
        
        # Pre-allocate generic range for bincounts to avoid dynamic resizing if possible
        self.max_embed_id = len(embeds) 

    def create_embeds_covers(self, embeds, same_embed_distance):
        n, d = embeds.shape
        index = faiss.IndexFlatL2(d)
        index.add(embeds)
        
        threshold = same_embed_distance ** 2
        lims, _, indices = index.range_search(embeds, threshold)
        
        embeds_covers = np.empty(n, dtype=object)
        
        # Optimize loop: Remove list comprehension, use numpy masking
        for i in range(n):
            start, end = lims[i], lims[i + 1]
            neighbors = indices[start:end]
            
            # Vectorized filtering for forward-looking covers only (j > i)
            # We sort strictly to allow searchsorted later
            future_neighbors = np.sort(neighbors[neighbors > i])
            embeds_covers[i] = future_neighbors
            
        return embeds_covers

    def initialize(self, capacity: int, index):
        self.items = {}
        self.curr_embed_id = 0
        super().initialize(capacity, index)

    def select_replacement_optimized(self, cand_id):
        """
        Algorithmic Optimization:
        Instead of calculating Union(All \ {x}) for every x (Quadratic complexity),
        we calculate global frequency counts.
        
        An item should be kept if it covers queries that NO ONE ELSE covers.
        We look for items that have the fewest "unique covers".
        """
        # 1. Gather keys: current cache + candidate
        # Converting keys to list is faster than separate key/value extraction
        active_ids = list(self.items.keys())
        active_ids.append(cand_id)
        
        # 2. Collect all valid future hits (covers) for these items
        # We process only valid covers > curr_embed_id
        valid_covers_map = {}
        all_future_hits = []
        
        curr = self.curr_embed_id
        
        # Batch collect future hits
        for eid in active_ids:
            # Get pre-computed covers
            covers = self.embeds_covers[eid]
            
            # Find subset of covers that are in the future
            # searchsorted is O(log N), much faster than checking all
            start_idx = covers.searchsorted(curr, side='right')
            
            if start_idx < len(covers):
                future = covers[start_idx:]
                valid_covers_map[eid] = future
                all_future_hits.append(future)
            else:
                # If an item has no future covers, it is the perfect eviction candidate.
                # Return immediately to save computation.
                return eid

        # 3. Flatten and Count Frequencies
        if not all_future_hits:
            return cand_id # No future hits for anyone, arbitrary eviction
            
        flat_hits = np.concatenate(all_future_hits)
        
        # bincount is extremely fast for integer counting
        # We need size=max_id+1. We can infer max from data or use self.max_embed_id
        counts = np.bincount(flat_hits, minlength=self.max_embed_id)
        
        # 4. Identify "Critical Hits" (Hits covered by exactly one cached item)
        # A hit is critical if count == 1.
        # We use a boolean mask where Index K is True if query K is covered by only 1 item.
        is_critical = (counts == 1)

        # 5. Score candidates
        # Score = How many critical hits does this candidate provide?
        # We want to EVICT the item with the LOWEST score (least unique contribution).
        min_score = float('inf')
        evict_id = cand_id
        
        for eid in active_ids:
            if eid not in valid_covers_map:
                return eid # Score is 0
            
            # Sum the boolean mask for this item's covers
            # This counts how many of this item's covers are 'critical'
            item_covers = valid_covers_map[eid]
            score = np.sum(is_critical[item_covers])
            
            if score < min_score:
                min_score = score
                evict_id = eid
                # Optimization: Can't get lower than 0
                if min_score == 0:
                    return evict_id
                    
        return evict_id

    def cache(self, embeds, embeds_ids, count_nn=1, texts=[]):
        # 1. Batch calculations for hits
        closest_dists, _ = self.get_closest_stored_embeds(embeds, count_nn)
        # Assuming get_closest_stored_embeds returns (n_samples, n_neighbors)
        # We check if ANY neighbor is within distance
        has_hit = np.any(closest_dists < self.same_embed_distance, axis=1)
        cache_hits = np.sum(has_hit)

        evicted_items = []
        rejected_items = []
        additions_embeds = []
        additions_ids = []

        # Update current time reference
        max_id = np.max(embeds_ids)
        self.curr_embed_id = max_id

        # 2. Process cache updates
        # We must process sequentially because cache state changes influence decisions
        for i, (embed, embed_id) in enumerate(zip(embeds, embeds_ids)):
            
            # If it's a hit, we technically don't need to do anything for LFU/OPT 
            # unless we need to refresh LRU positions (not needed here).
            # However, logic implies we insert only if space or better candidate exists.
            
            if embed_id in self.items:
                continue # Already in cache

            if len(self.items) < self.capacity:
                self.items[embed_id] = 1
                additions_embeds.append(embed)
                additions_ids.append(embed_id)
            else:
                # Use the optimized selector
                evict_id = self.select_replacement_optimized(embed_id)
                
                if evict_id == embed_id:
                    rejected_items.append(embed_id)
                else:
                    self.items.pop(evict_id)
                    evicted_items.append(evict_id)
                    self.items[embed_id] = 1
                    additions_embeds.append(embed)
                    additions_ids.append(embed_id)

        # 3. Batch updates to the index
        if additions_ids:
            self.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        if evicted_items:
            self.remove_ids(np.array(evicted_items))

        return cache_hits, evicted_items + rejected_items
class ClusterRelaxedOPT(Cache):
    def __init__(self, same_embed_distance, embeds, belady_boundary_coe=2.0):
        super().__init__(same_embed_distance)
        self.belady_boundary_coe = belady_boundary_coe
        self.embeds_covers, self.embeds_clusters = self.create_embeds_covers(embeds, same_embed_distance)

    def get_belady_boundary(self):
        return self.capacity * self.belady_boundary_coe

    def create_embeds_covers(self, embeds, same_embed_distance):
        embeds_clusters = cluster_complete_linkage_faiss(embeds, same_embed_distance)
        n_clusters = len(set(embeds_clusters))
        embeds_covers = [[] for _ in range(n_clusters)]
        for i_embed, i_cluster in enumerate(embeds_clusters):
            embeds_covers[i_cluster].append(i_embed)
        embeds_covers = [np.sort(embed_covers) for embed_covers in embeds_covers]
        return np.array(embeds_covers, dtype=object), embeds_clusters

    def initialize(self, capacity: int, index):
        self.items = {}
        self.curr_embed_id = 0
        self.belady_boundary = np.inf
        super().initialize(capacity, index)

    def get_next_hits(self, embeds_ids):
        curr_id = self.curr_embed_id
        next_hits = {}
        batch_rows = self.embeds_covers[embeds_ids]        
        for embed_id, row in zip(embeds_ids, batch_rows):
            i = row.searchsorted(curr_id, side='right')
            next_hit = np.inf
            if len(row) > 0 and len(row) > i:
                next_hit = row[i]
            next_hits[embed_id] = next_hit
        return next_hits
    
    def cache(self, embeds, embeds_ids, count_nn=1, texts=[]):
        closest_dists, _ = self.get_closest_stored_embeds(embeds, count_nn)
        cache_hits = np.sum(closest_dists < self.same_embed_distance, axis=1)
        evicted_items = []
        rejected_items = []
        additions = []
        self.curr_embed_id = max(embeds_ids)
        
        # Refresh next hits of stale items
        stale_items = [eid for (eid, next_hit) in self.items.items() if next_hit <= self.curr_embed_id]
        if len(stale_items) > 0:
            self.items.update(self.get_next_hits(stale_items))

        # Predict next hits
        embeds_next_hits = self.get_next_hits(embeds_ids)
        belady_boundary = self.get_belady_boundary()

        # Select candidates to evict
        evict_cands = [eid for eid, next_hit in self.items.items() if next_hit > self.curr_embed_id + belady_boundary]
        np.random.shuffle(evict_cands)

        for embed, embed_id in zip(embeds, embeds_ids):
            embed_next_hit = embeds_next_hits[embed_id]

            if self.capacity <= self.size():
                if evict_cands:
                    evicted_id = evict_cands.pop()
                else:
                    evicted_id = max(self.items, key=self.items.get, default=None)
                evicted_items.append(evicted_id)
                self.items.pop(evicted_id)

            if self.capacity > self.size():
                self.items[embed_id] = embed_next_hit
                additions.append((embed_id, embed))

        if additions:
            additions_embeds = [v for (_, v) in additions]
            additions_ids = [v for (v, _) in additions]
            self.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        if evicted_items:
            self.remove_ids(np.array(evicted_items))

        return cache_hits, evicted_items + rejected_items



class ClusterOPT(Cache):
    def __init__(self, same_embed_distance, embeds):
        super().__init__(same_embed_distance)
        self.embeds = np.asarray(embeds, dtype=np.float32)
        self.max_embed_id = len(embeds)
        self.curr_cover_counters = np.zeros(self.max_embed_id + 1, dtype=np.int32)
        self.embeds_covers, self.embeds_clusters, self.clusters_embeds = self.create_embeds_covers(self.embeds, same_embed_distance)

    def create_embeds_covers(self, embeds, same_embed_distance):
        n, d = embeds.shape
        index = faiss.IndexFlatL2(d)
        index.add(embeds)
        threshold = same_embed_distance ** 2
        lims, _, indices = index.range_search(embeds, threshold)
        embeds_covers = np.empty(n, dtype=object)
        for i in range(n):
            neighbors = indices[lims[i]:lims[i + 1]]
            future_neighbors = np.sort(neighbors[neighbors > i])
            embeds_covers[i] = future_neighbors
        embeds_clusters = cluster_complete_linkage_faiss(embeds, same_embed_distance)
        clusters_embeds = {}
        for i, v in enumerate(embeds_clusters):
            clusters_embeds.setdefault(v, []).append(i)
        
        return embeds_covers, embeds_clusters, clusters_embeds

    def initialize(self, capacity: int, index):
        self.items = {}
        self.curr_embed_id = 0
        # Reset counters
        self.curr_cover_counters.fill(0)
        super().initialize(capacity, index)

    def _update_counters_vectorized(self, ids, increment=True):
        if len(ids) == 0:
            return
        
        # Extract all relevant cover arrays at once
        selected_covers = self.embeds_covers[ids]
        
        if len(selected_covers) > 0:
            flat_indices = np.concatenate(selected_covers)
            if flat_indices.size > 0:
                val = 1 if increment else -1
                np.add.at(self.curr_cover_counters, flat_indices, val)

    def get_cluster_next_hits(self, embeds_ids):
        curr_id = self.curr_embed_id
        clusters_ids = self.embeds_clusters[embeds_ids]
        next_hits = {}
        for embed_id, cluster_id in zip(embeds_ids, clusters_ids):
            row = self.clusters_embeds[cluster_id]
            idx = np.searchsorted(row, curr_id, side='right')
            if idx < len(row):
                next_hits[embed_id] = row[idx]
            else:
                next_hits[embed_id] = np.inf
        return next_hits
    
    def get_next_hits(self, embeds_ids):
        curr_id = self.curr_embed_id
        next_hits = {}
        batch_rows = self.embeds_covers[embeds_ids]        
        for embed_id, row in zip(embeds_ids, batch_rows):
            i = row.searchsorted(curr_id, side='right')
            if i < len(row):
                next_hits[embed_id] = row[i]
            else:
                next_hits[embed_id] = np.inf
        return next_hits

    def get_combined_next_hits(self, embeds_ids):
        next_hits = self.get_next_hits(embeds_ids)
        cluster_next_hits = self.get_cluster_next_hits(embeds_ids)
        return {eid: (cluster_next_hits[eid], next_hits[eid]) for eid in embeds_ids}
    
    def cache(self, embeds, embeds_ids, count_nn=1, texts=[]):
        closest_dists, _ = self.get_closest_stored_embeds(embeds, count_nn)
        cache_hits = np.sum(closest_dists < self.same_embed_distance, axis=1)
        
        evicted_items = []
        rejected_items = []
        additions_embeds = []
        additions_ids = []
        
        self.curr_embed_id = max(embeds_ids)
        
        stale_items = [eid for (eid, (_, next_hit)) in self.items.items() if next_hit <= self.curr_embed_id]
        if stale_items:
            self.items.update(self.get_combined_next_hits(stale_items))
        
        embeds_next_hits = self.get_combined_next_hits(embeds_ids)
        for embed, embed_id in zip(embeds, embeds_ids):
            embed_next_hit_tuple = embeds_next_hits[embed_id] 
            max_next_hit_embed_id = max(self.items, key=self.items.get, default=None)
            max_next_hit_tuple = self.items.get(max_next_hit_embed_id, (float('inf'), float('inf')))
            target_hit = embed_next_hit_tuple[1]
            is_covered = False
            if target_hit != np.inf and target_hit < len(self.curr_cover_counters):
                is_covered = self.curr_cover_counters[int(target_hit)] > 0
            # Logic: Cache if we have space OR (Item is unique AND needed sooner than victim)
            should_cache = False
            if self.capacity > self.size():
                should_cache = True
            elif (not is_covered) and (embed_next_hit_tuple <= max_next_hit_tuple):
                should_cache = True
            if should_cache:
                if self.capacity <= self.size():
                    evicted_items.append(max_next_hit_embed_id)
                    self.items.pop(max_next_hit_embed_id)
                self.items[embed_id] = embed_next_hit_tuple
                additions_embeds.append(embed)
                additions_ids.append(embed_id)
            else: 
                rejected_items.append(embed_id)

        if additions_ids:
            self.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
            self._update_counters_vectorized(additions_ids, increment=True)
            
        if evicted_items:
            self.remove_ids(np.array(evicted_items))
            self._update_counters_vectorized(evicted_items, increment=False)
            
        return cache_hits, evicted_items + rejected_items

class RelaxedOPT(Cache):

    def __init__(self, same_embed_distance, embeds, belady_boundary_coe=2.0):
        super().__init__(same_embed_distance)
        self.embeds_covers = self.create_embeds_covers(embeds, same_embed_distance)
        self.belady_boundary_coe = belady_boundary_coe

    def get_belady_boundary(self):
        return self.capacity * self.belady_boundary_coe
    
    def create_embeds_covers(self, embeds, same_embed_distance):
        embeds = np.asarray(embeds, dtype=np.float32)
        n, d = embeds.shape
        index = faiss.IndexFlatL2(d)
        index.add(embeds)
        threshold = same_embed_distance ** 2
        lims, _, indices = index.range_search(embeds, threshold)
        embeds_covers = []
        for i in range(n):
            i_covers = np.sort(np.array([j for j in indices[lims[i]:lims[i + 1]] if j > i]))
            embeds_covers.append(i_covers)
        return np.array(embeds_covers, dtype=object)

    def initialize(self, capacity: int, index):
        self.items = {}
        self.belady_boundary = np.inf
        self.curr_embed_id = 0
        super().initialize(capacity, index)
    
    def get_next_hits(self, embeds_ids):
        curr_id = self.curr_embed_id
        next_hits = {}
        batch_rows = self.embeds_covers[embeds_ids]        
        for embed_id, row in zip(embeds_ids, batch_rows):
            i = row.searchsorted(curr_id, side='right')
            next_hit = np.inf
            if len(row) > 0 and len(row) > i:
                next_hit = row[i]
            next_hits[embed_id] = next_hit
        return next_hits
    
    def cache(self, embeds, embeds_ids, count_nn=1, texts=[]):
        closest_dists, _ = self.get_closest_stored_embeds(embeds, count_nn)
        cache_hits = np.sum(closest_dists < self.same_embed_distance, axis=1)
        evicted_items = []
        rejected_items = []
        additions = []
        self.curr_embed_id = max(embeds_ids)
        
        stale_items = [eid for (eid, next_hit) in self.items.items() if next_hit <= self.curr_embed_id]
        if len(stale_items) > 0:
            self.items.update(self.get_next_hits(stale_items))
        
        embeds_next_hits = self.get_next_hits(embeds_ids)
        belady_boundary = self.get_belady_boundary()
        evict_cands = [eid for eid, next_hit in self.items.items() if next_hit > self.curr_embed_id + belady_boundary]
        np.random.shuffle(evict_cands)
        for embed, embed_id in zip(embeds, embeds_ids):
            embed_next_hit = embeds_next_hits[embed_id] 
            if self.capacity <= self.size():
                if len(evict_cands) > 0:
                    evicted_eid = evict_cands.pop()
                else:
                    evicted_eid = max(self.items, key=self.items.get, default=None)
                evicted_items.append(evicted_eid)
                self.items.pop(evicted_eid)
            self.items[embed_id] = embed_next_hit
            additions.append((embed_id, embed))
        if additions:
            additions_embeds = [v for (_, v) in additions]
            additions_ids = [v for (v, _) in additions]
            self.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        if evicted_items:
            self.remove_ids(np.array(evicted_items))
        return cache_hits, evicted_items + rejected_items


def pad_array(arr, N, v):
    current_length = arr.shape[0]
    if current_length < N:
        pad_length = N - current_length
        return np.pad(arr, (pad_length, 0), mode='constant', constant_values=v)
    elif current_length > N:
        return arr[-N:]
    return arr


class RLB_Reg():

    def __init__(self, train_capacity, deltas_count=8, same_embed_distance=0.5, belady_boundary_coe=2, dim=384):
        self.train_capacity = train_capacity
        self.labeled_data = {}
        self.unlabeled_data = {}
        self.index_train = faiss.IndexIDMap2(faiss.IndexFlatL2(dim))
        self.reg = None
        self.train_counter = 0
        self.deltas_count = deltas_count
        self.same_embed_distance = same_embed_distance
        self.belady_boundary_coe = belady_boundary_coe
        self.dim = dim
    
    def get_train_counter(self):
        return self.train_counter
    
    def get_belady_boundary(self):
        return self.train_capacity * self.belady_boundary_coe
    
    def remove_labeled_from_training(self):
        self.labeled_data = {}

    def get_in_range_stored_embeds(self, embeds, radius):
        radius_squared = radius ** 2
        lims, dist2, ids = self.index_train.range_search(embeds, radius_squared)
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

    @staticmethod
    def calc_edc(deltas, edc_count):
        edcs = np.zeros(edc_count)
        for i_delta, delta in enumerate(deltas):
            if delta == -1:
                edcs[i_delta:] = -1
                break
            for edc_index in range(edc_count):
                decay_const = pow(2, 9 + edc_index + 1)
                decay_factor = pow(2, -delta / decay_const)
                edcs[edc_index] = 1 + edcs[edc_index] * decay_factor
        return edcs
    
    def record_for_label(self, embeds_ids, embeds, embeds_texts):
        if len(self.labeled_data) >= self.train_capacity:
            self.train()
        _, cache_hits = self.get_features(embeds_ids, embeds, embeds_texts)
        for embed_id, embed_cache_hits in cache_hits.items():
            for cache_hit_embed_id in embed_cache_hits:
                entry = self.unlabeled_data.pop(cache_hit_embed_id, None)
                if entry is not None:
                    elapsed_time = embed_id - entry[1]
                    entry[1] = np.log1p(elapsed_time)
                    self.labeled_data[cache_hit_embed_id] = entry
        self.index_train.add_with_ids(embeds, np.array(embeds_ids))
        if self.index_train.ntotal > self.train_capacity:
            count_remove = self.index_train.ntotal - self.train_capacity
            begin = max(embeds_ids) - self.train_capacity
            embeds_ids_remove = list(range(begin, begin + count_remove)) 
            self.index_train.remove_ids(np.array(embeds_ids_remove))
        
        curr_time = max(embeds_ids)
        old_unlabeled = [v for v in self.unlabeled_data.keys() if (curr_time - v) >= self.train_capacity] 
        for embed_id in old_unlabeled:
            entry = self.unlabeled_data.pop(embed_id)
            entry[1] = np.log1p(2 * self.train_capacity)
            self.labeled_data[embed_id] = entry

    
    
    def record_for_training(self, embeds_ids, embeds, embeds_texts, curr_time):
        features, cache_hits = self.get_features(embeds_ids, embeds, embeds_texts)
        for embed_id, embed_cache_hits in cache_hits.items():
            self.unlabeled_data[embed_id] = [features.loc[embed_id], curr_time]
        

            
    def predict_tmp(self, embeds_ids, embeds, embeds_texts, actual_log):
        # 1) feature extraction
        X, _ = self.get_features(embeds_ids, embeds, embeds_texts)       
        log_time_hit_pred = self.reg.predict(X)

        # 2) predict in log‑space, then convert to real time
        log_pred = log_time_hit_pred               # log1p(time_to_hit̂ )
        t_pred   = np.expm1(log_pred)                   # time_to_hit̂  in original units
        y_pred   = embeds_ids + t_pred                  # absolute time of next hit

        # 3) diagnostics ────────────────
        # (i) MAE in log‑space (only where actual is uncensored)
        mask = np.isfinite(actual_log)
        if mask.any():
            log_mae = np.mean(np.abs(actual_log[mask] - log_pred[mask]))
            print(f"Log‑space MAE (uncensored) : {log_mae:.4f}")

            # (ii) MAE in original units
            mae_orig = np.mean(np.abs(np.expm1(actual_log[mask]) - t_pred[mask]))
            print(f"Original‑time MAE          : {mae_orig:.4f}")

        return y_pred
    
    def predict(self, embeds_ids, embeds, embeds_texts):
        X, _ = self.get_features(embeds_ids, embeds, embeds_texts)       
        log_time_hit_pred = self.reg.predict(X)
        time_hit_pred = np.expm1(log_time_hit_pred)
        y = embeds_ids + time_hit_pred 
        return y
    
    def get_features(self, embeds_ids, embeds, embeds_text):
        dists, ids = self.get_in_range_stored_embeds(embeds, self.same_embed_distance**2)
        
        features = []
        cache_hits = {}
        
        for i, (embed_id, embed_text) in enumerate(zip(embeds_ids, embeds_text)):
            hits_dists = dists[i]
            hits_ids = list(ids[i])
            if max(hits_ids, default=-1) > embed_id:
                x = 3
            if embed_id not in hits_ids:
                hits_ids.append(embed_id)
            sorted_hits = np.sort(hits_ids)
            deltas = np.diff(sorted_hits)
            edc = self.calc_edc(deltas, 4)
            recent_deltas = pad_array(deltas, self.deltas_count, -1)
            count_chars = len(embed_text)
            count_whitespace = sum(ch.isspace() for ch in embed_text)
            words = embed_text.split()
            count_words = len(words)
            count_vocab = len(set(words))
            count_sents = len(re.split(r"[.!?;]+", embed_text))
            mean_word_len = sum([len(word) for word in words], 0) / len(words) if len(words) > 0 else 0
            index_in_hits = list(reversed(sorted_hits)).index(embed_id)
            embed_features = {
                "embed_id": embed_id,
                # "count hits": embed_dists.size,
                "mean hit distance": np.mean(hits_dists) if len(hits_dists) > 0 else -1,
                # Misc
                # "data available": data_available,
                "index in hits": index_in_hits,
                # Deltas related
                # Sentence related
                #mean_word_len,
                #count_sents,
                #count_vocab,
                #"count words": count_words,
                #count_whitespace,
                #"count chars": count_chars,
                #"surprisal": calculate_surprisal(embed_text),
                # Distance related
                #np.min(embed_dists) if embed_dists.size else -1,
                #np.max(embed_dists) if embed_dists.size else -1,
                #".25 quantile distance": np.quantile(embed_dists, .25) if embed_dists.size else -1,
                ".25 quantile distance": np.quantile(hits_dists, .25) if hits_dists.size else -1,
                ".75 quantile distance": np.quantile(hits_dists, .75) if hits_dists.size else -1,
                "std hit distance": np.std(hits_dists) if hits_dists.size else -1,
            }
            # add delta
            for i, delta in enumerate(recent_deltas):
                embed_features[f'delta_{i}'] = delta
            for i, edc in enumerate(edc):
                embed_features[f'edc_{i}'] = edc

            features.append(embed_features)
            cache_hits[embed_id] = hits_ids

        features = pd.DataFrame(features).set_index('embed_id').fillna(-1).astype(np.float32)
        return features, cache_hits

    def train(self):
        self.train_counter += 1
        # print("Training...")

        # Extract features and labels
        features_list = []
        labels_list = []

        for features_df, label in self.labeled_data.values():
            features_list.append(features_df)  # each is a 1-row DataFrame
            labels_list.append(label)

        # Concatenate all features into a single DataFrame
        X = pd.DataFrame(features_list)
        y = pd.Series(labels_list, name="target")

        tscv = TimeSeriesSplit(n_splits=5)
        fold = 0
        scores = []

        for train_index, test_index in tscv.split(X):
            fold += 1
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            reg = xgb.XGBRegressor(
                max_depth=4,
                n_estimators=200,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=2.0,
                objective='reg:squarederror',
                verbosity=0,
                random_state=42
            )

            reg.fit(X_train, y_train)

            preds_train = reg.predict(X_train)
            preds_test = reg.predict(X_test)

            mse_train = mean_squared_error(y_train, preds_train)
            mae_train = mean_absolute_error(y_train, preds_train)
            r2_train = r2_score(y_train, preds_train)

            mse_test = mean_squared_error(y_test, preds_test)
            mae_test = mean_absolute_error(y_test, preds_test)
            r2_test = r2_score(y_test, preds_test)

            print(f"\nFold {fold}")
            print(f"Train    | MSE {mse_train:.4f}  MAE {mae_train:.4f}  R² {r2_train:.4f}")
            print(f"Test     | MSE {mse_test:.4f}  MAE {mae_test:.4f}  R² {r2_test:.4f}")
            scores.append((mse_test, mae_test, r2_test))

        # Optional: Train final model on all data
        self.reg = xgb.XGBRegressor(
            max_depth=4,
            n_estimators=200,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            objective='reg:squarederror',
            verbosity=0,
            random_state=42
        )
        self.reg.fit(X, y)

        # Final feature importances
        imp = self.reg.get_booster().get_score(importance_type='gain')
        print("\nFinal Feature Importances (Gain):")
        for k, v in sorted(imp.items(), key=lambda x: x[1], reverse=True):
            print(f"{k:25s} → Gain: {v:.3f}")
        
        self.labeled_data = {}
        
        self.remove_labeled_from_training()


    
class RelaxedLearnedOPT(Cache):

    def __init__(self, same_embed_distance, deltas_count=4, train_capacity_ratio=1.0, belady_boundary_coe=2.0, dim=384):
        super().__init__(same_embed_distance)
        self.dim = dim
        self.belady_boundary_coe = belady_boundary_coe
        self.deltas_count = deltas_count
        self.train_capacity_ratio = train_capacity_ratio

    def initialize(self, capacity: int, index):
        self.train_counter = 0
        self.items = OrderedDict()
        self.belady_boundary = np.inf
        self.curr_embed_id = 0
        train_capacity = int(self.train_capacity_ratio * capacity)
        self.reg = RLB_Reg(train_capacity, self.deltas_count, self.same_embed_distance, self.belady_boundary_coe, self.dim)
        self.index_train = faiss.IndexIDMap2(faiss.IndexFlatL2(self.dim))
        super().initialize(capacity, index)    
    
    def predict(self, embeds_ids, embeds, embeds_text):
        return self.reg.predict(np.array(embeds_ids), np.array(embeds), embeds_text)
    
    def ml_evict_embed(self):
        predict_size = min(64, len(self.items))
        sampled_items = random.sample(list(self.items.items()), predict_size)
        embeds_ids, embeds_and_embeds_texts = zip(*sampled_items)
        (embeds, embeds_text) = zip(*embeds_and_embeds_texts)
        predicts = self.reg.predict(np.array(embeds_ids), np.array(embeds), embeds_text)
        index = np.argmin(np.array(predicts))
        return embeds_ids[index]
    
    def get_evict_embed_id(self):
        if self.reg.get_train_counter() > 0:
            evict_embed_id = self.ml_evict_embed()
            embed, embed_text = self.items.pop(evict_embed_id)
            return evict_embed_id
        else:
            evict_embed_id, (embed, embed_text) = self.items.popitem(last=False)
            return evict_embed_id

    def sample_for_recording(self, count_record: int) -> None:
        cands = list(set(self.items.keys())-set(self.reg.unlabeled_data.keys()))
        embeds_ids = random.sample(cands, count_record)
        embeds_data = [self.items[embed_id] for embed_id in embeds_ids]
        embeds, embeds_texts = zip(*embeds_data)
        self.reg.record_for_training(np.array(embeds_ids), np.array(embeds), embeds_texts, self.curr_embed_id)
        
    
    def cache(self, embeds, embeds_ids, count_nn=1, texts=[]):
        closest_dists, cache_hits_ids = self.get_closest_stored_embeds(embeds, count_nn)
        self.curr_embed_id = max(embeds_ids)
        mask = closest_dists < self.same_embed_distance
        cache_hits_indices = np.where(mask)
        for i_embed, i_nn in zip(*cache_hits_indices):
            nn = cache_hits_ids[i_embed][i_nn]
            self.items.move_to_end(nn)
        
        # For maintaining LRU
        if self.reg.train_counter == 0:
            flat_cache_hits_ids = cache_hits_ids.ravel()[cache_hits_ids.ravel() != -1]
            for hit_embed_id in flat_cache_hits_ids:
                self.items.move_to_end(hit_embed_id)
        
        cache_hits = np.sum(closest_dists < self.same_embed_distance, axis=1)
        evicted_items = []
        rejected_items = []
        additions = []
        
        for embed_id, embed, embed_text in zip(embeds_ids, embeds, texts): 
            if self.capacity <= self.size():
                evict_embed_id = self.get_evict_embed_id()
                evicted_items.append(evict_embed_id)
            if self.capacity > self.size():
                self.items[embed_id] = (embed, embed_text)
                additions.append((embed_id, embed))

        self.reg.record_for_label(np.array(embeds_ids), np.array(embeds), texts)
        self.sample_for_recording(len(embeds))
                
        if additions:
            additions_embeds = [v for (_, v) in additions]
            additions_ids = [v for (v, _) in additions]
            self.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        if evicted_items:
            self.remove_ids(np.array(evicted_items))
        return cache_hits, evicted_items + rejected_items

if __name__ == "__main__":
    cluster0 = list(np.linspace(0, 1, 100))
    cluster1 = list(np.linspace(3, 4, 100))
    embeds = [7] + random.shuffle(cluster0 + cluster1) + [7.1] 
    index_opt = NaiveVectorStore(dim = 1)
    index_copt = NaiveVectorStore(dim = 1)
    embeds = None
    opt = OPT(1, embeds)
    opt.initialize(2, index_opt)
    copt = ClusterOPT(1, embeds)
    copt.initialize(2, index_copt)
    for i_embed, embed in enumerate(embeds):
        res_copt = copt.cache(np.array([embed]), list(i_embed))
        res_opt = opt.cache(np.array([embed]), list(i_embed))
        x = 3
        
        
    