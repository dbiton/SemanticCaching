import bisect
from collections import OrderedDict
import pandas as pd
from cache import Cache
import numpy as np
import faiss
import lightgbm as lgb
import random

class OPT(Cache):

    def __init__(self, same_embed_distance, embeds):
        super().__init__(same_embed_distance)
        self.embeds_covers = self.create_embeds_covers(embeds, same_embed_distance)

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
        self.curr_embed_id = 0
        super().initialize(capacity, index)
    
    def get_next_hits(self, embeds_ids):
        curr_id = self.curr_embed_id
        next_hits = {}
        for embed_id, row in zip(embeds_ids, self.embeds_covers[embeds_ids]):
            i = row.searchsorted(curr_id, side='right')
            next_hit = np.inf
            if len(row) > 0 and len(row) > i:
                next_hit = row[i]
            next_hits[embed_id] = next_hit
        return next_hits
    
    def request(self, embeds, embeds_ids, count_nn=1):
        closest_dists, _ = self.get_closest_stored_embeds(embeds, count_nn)
        cache_hits = np.sum(closest_dists < self.same_embed_distance, axis=1)
        evicted_items = []
        rejected_items = []
        additions = []
        
        stale_items = [eid for (eid, next_hit) in self.items.items() if next_hit < self.curr_embed_id]
        if len(stale_items) > 0:
            self.items.update(self.get_next_hits(stale_items))
        
        embeds_next_hits = self.get_next_hits(embeds_ids)
        for embed, embed_id in zip(embeds, embeds_ids):
            embed_next_hit = embeds_next_hits[embed_id] 
            self.curr_embed_id = embed_id
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


class RelaxedOPT(Cache):
    def __init__(self, same_embed_distance, embeds, belady_boundary_coe=2.0):
        super().__init__(same_embed_distance)
        self.embeds_covers = self.create_embeds_covers(embeds, same_embed_distance)
        self.belady_boundary_coe=belady_boundary_coe

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
        for embed_id, row in zip(embeds_ids, self.embeds_covers[embeds_ids]):
            i = row.searchsorted(curr_id, side='right')
            next_hit = np.inf
            if len(row) > 0 and len(row) > i:
                next_hit = row[i]
            next_hits[embed_id] = next_hit
        return next_hits
    
    def request(self, embeds, embeds_ids, count_nn=1):
        closest_dists, _ = self.get_closest_stored_embeds(embeds, count_nn)
        cache_hits = np.sum(closest_dists < self.same_embed_distance, axis=1)
        evicted_items = []
        rejected_items = []
        additions = []
        
        stale_items = [eid for (eid, next_hit) in self.items.items() if next_hit < self.curr_embed_id]
        if len(stale_items) > 0:
            self.items.update(self.get_next_hits(stale_items))
        
        embeds_next_hits = self.get_next_hits(embeds_ids)
        belady_boundary = self.get_belady_boundary()
        evict_cands = [eid for eid, next_hit in self.items.items() if next_hit > self.curr_embed_id + belady_boundary]
        np.random.shuffle(evict_cands)
        for embed, embed_id in zip(embeds, embeds_ids):
            embed_next_hit = embeds_next_hits[embed_id] 
            self.curr_embed_id = embed_id
            if self.capacity <= self.size():
                if len(evict_cands) > 0:
                    evicted_eid = evict_cands.pop()
                else:
                    evicted_eid = max(self.items, key=self.items.get, default=None)
                evicted_items.append(evicted_eid)
                self.items.pop(evicted_eid, None)
            self.items[embed_id] = embed_next_hit
            additions.append((embed_id, embed))
        if additions:
            additions_embeds = [v for (_, v) in additions]
            additions_ids = [v for (v, _) in additions]
            self.index.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        if evicted_items:
            self.index.remove_ids(np.array(evicted_items))
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
    def __init__(self, train_capacity, deltas_count = 8, same_embed_distance = 0.5, belady_boundary_coe = 2, dim=384):
        self.train_capacity = train_capacity
        self.training_data = {}
        self.index_train = faiss.IndexIDMap2(faiss.IndexFlatL2(dim))
        self.reg = None
        self.deltas_count = deltas_count
        self.same_embed_distance = same_embed_distance
        self.labeled_count = 0
        self.belady_boundary_coe = belady_boundary_coe
        self.dim = dim
    
    def is_trained(self):
        return self.reg is not None
    
    def get_belady_boundary(self):
        return self.train_capacity * self.belady_boundary_coe
    
    def get_default_label(self):
        return np.log2(self.get_belady_boundary() * 2)
    
    def train(self):
        print("Training...")
        self.reg = lgb.LGBMRegressor()
        data = pd.DataFrame(self.training_data.values())
        X = np.array(data[0].tolist()).astype(np.float32)
        default_label = self.get_default_label()
        y = data[1].fillna(default_label).astype(np.float32).to_list()
        self.reg.fit(X, y)
        self.remove_labeled_from_training()
    
    def remove_labeled_from_training(self):
        removed_embeds_ids = [eid for (eid, entry) in self.training_data.items() if entry[1] is not None]
        self.training_data = {k: v for (k, v) in self.training_data.items() if k not in removed_embeds_ids}
        self.index_train.remove_ids(np.array(removed_embeds_ids))
        self.labeled_count = 0

    @staticmethod
    def calc_edc(deltas, edc_count):
        edcs = np.zeros(edc_count)
        for i_delta, delta in enumerate(deltas):
            if delta == -1:
                edcs[i_delta:] = -1
                break
            for edc_index in range(edc_count):
                decay_const = pow(2, 9 + edc_index + 1)
                decay_factor = pow(2, - delta / decay_const)
                edcs[edc_index] = 1 + edcs[edc_index] * decay_factor
        return edcs
    
    def record_for_training(self, embeds_ids, embeds):
        if self.labeled_count >= self.train_capacity:
            self.train()
        features, cache_hits = self.get_features(embeds_ids, embeds)
        for embed_id, embed_cache_hits in cache_hits.items():
            self.training_data[embed_id] = [features[embed_id], None]
            for cache_hit_embed_id in embed_cache_hits:
                entry = self.training_data.get(cache_hit_embed_id, None)
                if entry is not None and entry[1] is None:
                    self.labeled_count += 1
                    entry[1] = np.log2(embed_id - cache_hit_embed_id)
        self.index_train.add_with_ids(embeds, np.array(embeds_ids))
    
    def predict(self, embeds_ids, embeds):
        features, _ = self.get_features(embeds_ids, embeds)
        X = np.array(list(features.values()))
        return self.reg.predict(X)
    
    def get_features(self, embeds_ids, embeds):
        dists, ids = self.index_train.search(embeds, self.deltas_count)
        dists = np.sqrt(dists)
        
        features = {}
        cache_hits = {}

        for i, embed_id in enumerate(embeds_ids):
            embed_dists = dists[i]
            embed_ids = ids[i]

            valid_mask = embed_ids != -1
            valid_ids = embed_ids[valid_mask]
            valid_dists = embed_dists[valid_mask]

            hits_mask = valid_dists <= self.same_embed_distance
            hits_ids = valid_ids[hits_mask]
            hits_dists = valid_dists[hits_mask]

            reasonable_mask = valid_dists <= 1e10
            reasonable_dists = valid_dists[reasonable_mask]

            if hits_ids.size > 0:
                sorted_hits = np.sort(hits_ids)
                deltas = np.diff(sorted_hits)
            else:
                deltas = np.array([], dtype=int)
            deltas = pad_array(deltas, self.deltas_count, -1)
            edc = self.calc_edc(deltas, self.deltas_count)

            curr_features = np.hstack((
                np.mean(reasonable_dists) if reasonable_dists.size else -1,
                np.std(reasonable_dists) if reasonable_dists.size else -1,
                np.mean(hits_dists) if hits_dists.size else -1,
                np.std(hits_dists) if hits_dists.size else -1,
                deltas,
                edc,
                hits_ids.size,
            ))

            features[embed_id] = np.nan_to_num(curr_features, nan=-1)
            cache_hits[embed_id] = hits_ids

        return features, cache_hits
    
    
class RelaxedLearnedOPT(Cache):
    def __init__(self, same_embed_distance, deltas_count=4, train_capacity_ratio=1.0, belady_boundary_coe = 2.0, dim = 384):
        super().__init__(same_embed_distance)
        self.dim = dim
        self.belady_boundary_coe = belady_boundary_coe
        self.deltas_count = deltas_count
        self.train_capacity_ratio = train_capacity_ratio 

    def initialize(self, capacity: int, index):
        self.first_fit = False
        self.items = []
        self.labeled_count = 0
        self.belady_boundary = np.inf
        self.curr_embed_id = 0
        train_capacity = int(self.train_capacity_ratio * capacity)
        self.reg = RLB_Reg(train_capacity, self.deltas_count, self.same_embed_distance, self.belady_boundary_coe, self.dim)
        self.training_data = {}
        self.index_train = faiss.IndexIDMap2(faiss.IndexFlatL2(self.dim))
        super().initialize(capacity, index)    
    
    def predict(self, embeds_ids, embeds):
        y = self.reg.predict(np.array(embeds_ids), np.array(embeds))
        return 2 ** y + embeds_ids
    
    def request(self, embeds, embeds_ids, count_nn=1):
        closest_dists, _ = self.get_closest_stored_embeds(embeds, count_nn)
        self.reg.record_for_training(embeds_ids, embeds)
        cache_hits = np.sum(closest_dists < self.same_embed_distance, axis=1)
        evicted_items = []
        rejected_items = []
        additions = []
        
        if self.reg.is_trained():
            if not self.first_fit:
                self.first_fit = True
                _, all_embeds_ids, all_embeds = zip(*self.items)
                all_next_hits = self.predict(all_embeds_ids, all_embeds)
                self.items = sorted(list(zip(all_next_hits, all_embeds_ids, all_embeds)))
            next_hits = self.predict(embeds_ids, embeds)
            entries = sorted(list(zip(next_hits, embeds_ids, embeds)))
        else:
            entries = list(zip(list([-1] * len(embeds)), embeds_ids, embeds))
        
        for next_hit, embed_id, embed in entries:            
            if self.capacity <= self.size():
                # remove worst
                max_hit, max_embed_id, max_embed = self.items[-1]
                if max_hit >= next_hit:
                    self.items.pop()
                    evicted_items.append(max_embed_id)
                else:
                    evicted_items.append(embed_id)
            if self.capacity >= self.size():
                bisect.insort(self.items, (next_hit, embed_id, embed))
                additions.append((embed_id, embed))
            self.curr_embed_id += 1

        if additions:
            additions_embeds = [v for (_, v) in additions]
            additions_ids = [v for (v, _) in additions]
            self.index.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        if evicted_items:
            self.index.remove_ids(np.array(evicted_items))
        return cache_hits, evicted_items + rejected_items
