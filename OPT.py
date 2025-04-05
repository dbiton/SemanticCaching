from collections import OrderedDict
import pandas as pd
from cache import Cache
import numpy as np
import faiss
from xgboost.sklearn import XGBRegressor
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

class RelaxedLearnedOPT(Cache):
    def __init__(self, same_embed_distance, deltas_count=4, train_capacity_ratio=1.0, belady_boundary_coe = 2.0, dim = 384):
        super().__init__(same_embed_distance)
        self.dim = dim
        self.belady_boundary_coe = belady_boundary_coe
        self.deltas_count = deltas_count
        self.train_capacity_ratio = train_capacity_ratio 

    def get_belady_boundary(self):
        return self.capacity * self.belady_boundary_coe

    def initialize(self, capacity: int, index):
        self.items = OrderedDict()
        self.belady_boundary = np.inf
        self.curr_embed_id = 0
        self.reg = None
        self.train_capacity = int(self.train_capacity_ratio * capacity)
        self.training_data = {}
        self.index_train = faiss.IndexIDMap2(faiss.IndexFlatL2(self.dim))
        super().initialize(capacity, index)
    
    def train_reg(self):
        self.reg = XGBRegressor(objective='reg:squarederror', verbosity="2", random_state=42)
        data = pd.DataFrame(self.training_data.values())
        X = np.array(data['features'].tolist())
        y = data['label'].to_list()
        self.reg.fit(X, y)
        self.index_train = faiss.IndexIDMap2(faiss.IndexFlatL2(self.dim))
        self.training_data = {}
    
    def get_features(self, embeds_ids, embeds, with_labels=False):
        threshold = self.same_embed_distance ** 2
        lims, distances, indices = self.index_train.range_search(embeds, threshold)
        result = {}
        for i, embed_id in enumerate(embeds_ids):
            start = lims[i]
            end = lims[i + 1]
            hits = pad_array(indices[start:end], self.deltas_count, -1)
            relative_hits = np.where(hits != -1, embed_id - hits, hits)
            dists = pad_array(distances[start:end], self.deltas_count, -1)
            result[embed_id] = {"features": np.hstack((relative_hits, dists))}
            if with_labels:
                result[embed_id]["label"] = np.log(self.train_capacity * 2)
                for i in indices[start:end]:
                    self.training_data[i]['label'] = min(self.training_data[i]['label'], np.log(embed_id - i)) 
        return result
      
    def record_for_training(self, embeds_ids, embeds):
        self.training_data.update(self.get_features(embeds_ids, embeds, with_labels=True))
        self.index_train.add_with_ids(embeds, np.array(embeds_ids))
    
    def evict(self):
        if self.reg:
            batch_size = 16
            if len(self.items) > batch_size:
                indice = random.sample(range(len(self.items)), batch_size)
            else:
                indice = list(range(len(self.items)))
            embeds_ids, embeds = np.array(list(self.items.keys()))[indice], np.array(list(self.items.values()))[indice]
            data = self.get_features(embeds_ids, embeds, with_labels=False)
            data = pd.DataFrame(data.values())
            X = np.array(data['features'].tolist())
            y = self.reg.predict(X)
            cands = np.where(y < np.log(self.get_belady_boundary()))
            if y.std() > 0:
                x = 3
            if len(cands) > 0:
                evicted_eid = embeds_ids[np.random.choice(cands[0])]
            else:
                evicted_eid = embeds_ids[np.random.choice(embeds_ids)]
            self.items.pop(evicted_eid)
            return evicted_eid
        else:
            return self.items.popitem(last=False)[0]
            
    
    def request(self, embeds, embeds_ids, count_nn=1):
        closest_dists, _ = self.get_closest_stored_embeds(embeds, count_nn)
        cache_hits = np.sum(closest_dists < self.same_embed_distance, axis=1)
        self.record_for_training(embeds_ids, embeds)
        if len(self.training_data) == self.train_capacity:
            self.train_reg()
        evicted_items = []
        rejected_items = []
        additions = []
        for embed, embed_id in zip(embeds, embeds_ids):
            if self.capacity <= self.size():
                evicted_eid = self.evict()
                evicted_items.append(evicted_eid)
            self.items[embed_id] = embed
            additions.append((embed_id, embed))
            self.curr_embed_id += 1
        # boilerplate
        if additions:
            additions_embeds = [v for (_, v) in additions]
            additions_ids = [v for (v, _) in additions]
            self.index.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        if evicted_items:
            self.index.remove_ids(np.array(evicted_items))
        return cache_hits, evicted_items + rejected_items
