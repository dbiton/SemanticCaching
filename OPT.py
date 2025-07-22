import bisect
from collections import OrderedDict
import pandas as pd
from cache import Cache
import numpy as np
import faiss
import lightgbm as lgb
import xgboost as xgb
import random
from scipy import stats

from freq_reg import FreqReg
from surprisal.estimate_frequency import calculate_surprisal
import re
from scipy.stats._stats_py import median_abs_deviation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


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
    
    def request(self, embeds, embeds_ids, count_nn=1, texts=[]):
        closest_dists, _ = self.get_closest_stored_embeds(embeds, count_nn)
        cache_hits = np.sum(closest_dists < self.same_embed_distance, axis=1)
        evicted_items = []
        rejected_items = []
        additions = []
        
        stale_items = [eid for (eid, next_hit) in self.items.items() if next_hit <= self.curr_embed_id]
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
        for embed_id, row in zip(embeds_ids, self.embeds_covers[embeds_ids]):
            i = row.searchsorted(curr_id, side='right')
            next_hit = np.inf
            if len(row) > 0 and len(row) > i:
                next_hit = row[i]
            next_hits[embed_id] = next_hit
        return next_hits
    
    def request(self, embeds, embeds_ids, count_nn=1, texts=[]):
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

    def __init__(self, train_capacity, deltas_count=8, same_embed_distance=0.5, belady_boundary_coe=2, dim=384):
        self.train_capacity = train_capacity
        self.training_data = {}
        self.index_train = faiss.IndexIDMap2(faiss.IndexFlatL2(dim))
        self.reg = None
        self.train_counter = 0
        self.deltas_count = deltas_count
        self.same_embed_distance = same_embed_distance
        self.labeled_count = 0
        self.awaiting_label_embed_ids = set()
        self.belady_boundary_coe = belady_boundary_coe
        self.dim = dim
    
    def get_train_counter(self):
        return self.train_counter
    
    def get_belady_boundary(self):
        return self.train_capacity * self.belady_boundary_coe
    
    def remove_labeled_from_training(self):
        removed_embeds_ids = [eid for (eid, entry) in self.training_data.items() if entry[1] is not None]
        self.training_data = {k: v for (k, v) in self.training_data.items() if k not in removed_embeds_ids}
        self.index_train.remove_ids(np.array(removed_embeds_ids))
        self.labeled_count = 0

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
    
    def record_for_training(self, embeds_ids, embeds, embeds_texts):
        if self.labeled_count >= self.train_capacity:
            self.train()
        features, cache_hits = self.get_features(embeds_ids, embeds, embeds_texts)
        for embed_id, embed_cache_hits in cache_hits.items():
            self.training_data[embed_id] = [features.loc[embed_id], None]
            self.awaiting_label_embed_ids.add(embed_id)
            for cache_hit_embed_id in embed_cache_hits:
                entry = self.training_data.get(cache_hit_embed_id, None)
                if entry is not None and entry[1] is None:
                    self.labeled_count += 1
                    self.awaiting_label_embed_ids.remove(cache_hit_embed_id)
                    entry[1] = np.log1p(embed_id - cache_hit_embed_id)
        
        curr_time = max(embeds_ids)
        old_unlabeled = [v for v in self.awaiting_label_embed_ids if (curr_time - v) >= self.get_belady_boundary()] 
        for embed_id in old_unlabeled:
            entry = self.training_data.get(embed_id)
            entry[1] = np.log1p(self.get_belady_boundary())
            self.labeled_count += 1
        self.awaiting_label_embed_ids.difference_update(old_unlabeled)
        
        self.index_train.add_with_ids(embeds, np.array(embeds_ids))
    
    def predict_tmp(self, embeds_ids, embeds, embeds_texts, actual):
        X, _ = self.get_features(embeds_ids, embeds, embeds_texts)       
        log_time_hit_pred = self.reg.predict(X)
        time_hit_pred = np.expm1(log_time_hit_pred)
        y = embeds_ids + time_hit_pred 
        print(np.mean(np.abs(actual - log_time_hit_pred)))
        return y
    
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
            embed_dists = dists[i]
            embed_ids = ids[i]
            sorted_hits = np.sort([embed_id] + embed_ids)
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
            data_available = len(self.training_data)
            embed_features = {
                "embed_id": embed_id,
                "count hits": embed_dists.size,
                "mean hit distance": np.mean(embed_dists) if len(embed_dists) > 0 else -1,
                # Misc
                "data available": data_available,
                # Deltas related
                # edc,
                # Sentence related
                #mean_word_len,
                #count_sents,
                #count_vocab,
                "count words": count_words,
                #count_whitespace,
                "count chars": count_chars,
                "surprisal": calculate_surprisal(embed_text),
                # Distance related
                #np.min(embed_dists) if embed_dists.size else -1,
                #np.max(embed_dists) if embed_dists.size else -1,
                ".05 distance": np.quantile(embed_dists, .05) if embed_dists.size else -1,
                #np.quantile(embed_dists, .25) if embed_dists.size else -1,
                #np.quantile(embed_dists, .75) if embed_dists.size else -1,
                ".95 quantile distance": np.quantile(embed_dists, .95) if embed_dists.size else -1,
                "std hit distance": np.std(embed_dists) if embed_dists.size else -1,
            }
            # add delta
            for i, delta in enumerate(recent_deltas):
                embed_features[f'delta_{i}'] = delta

            features.append(embed_features)
            cache_hits[embed_id] = embed_ids

        features = pd.DataFrame(features).set_index('embed_id').fillna(-1).astype(np.float32)
        return features, cache_hits
    
    def train(self):
        self.train_counter += 1
        #print("Training...")

        # Extract features and labels
        features_list = []
        labels_list = []

        for features_df, label in self.training_data.values():
            if label is None:
                continue
            features_list.append(features_df)  # each is a 1-row DataFrame
            labels_list.append(label)

        # Concatenate all features into a single DataFrame
        X = pd.DataFrame(features_list)
        y = pd.Series(labels_list, name="target")
    
        self.reg = xgb.XGBRegressor(max_depth=4, n_estimators=256, verbosity=0, objective='reg:squarederror')

        # Split into train (80%) and test (20%) sets
        split_index = int(len(X) * 0.95)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Train on training set only
        self.reg.fit(X_train, y_train)

        # Evaluate on training set
        preds_train = self.reg.predict(X_train)
        mse_train = mean_squared_error(y_train, preds_train)
        mae_train = mean_absolute_error(y_train, preds_train)
        r2_train = r2_score(y_train, preds_train)

        print(f"Training MSE: {mse_train:.4f}")
        print(f"Training MAE: {mae_train:.4f}")
        print(f"Training R²: {r2_train:.4f}")

        # Evaluate on test set
        preds_test = self.reg.predict(X_test)
        mse_test = mean_squared_error(y_test, preds_test)
        mae_test = mean_absolute_error(y_test, preds_test)
        r2_test = r2_score(y_test, preds_test)

        print(f"Test MSE: {mse_test:.4f}")
        print(f"Test MAE: {mae_test:.4f}")
        print(f"Test R²: {r2_test:.4f}")
        
        imp = self.reg.get_booster().get_score(importance_type='gain')
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
        self.evict_cands = set()
        self.items = OrderedDict()
        self.labeled_count = 0
        self.belady_boundary = np.inf
        self.curr_embed_id = 0
        train_capacity = int(self.train_capacity_ratio * capacity)
        self.reg = RLB_Reg(train_capacity, self.deltas_count, self.same_embed_distance, self.belady_boundary_coe, self.dim)
        self.training_data = {}
        self.index_train = faiss.IndexIDMap2(faiss.IndexFlatL2(self.dim))
        super().initialize(capacity, index)    
    
    def predict(self, embeds_ids, embeds, embeds_text):
        return self.reg.predict(np.array(embeds_ids), np.array(embeds), embeds_text)
    
    def select_evict_cands(self):
        predict_size = min(64, len(self.items))
        evict_size = 1
        sampled_items = random.sample(list(self.items.items()), predict_size)
        embeds_ids, embeds_and_embeds_texts = zip(*sampled_items)
        (embeds, embeds_text) = zip(*embeds_and_embeds_texts)
        predicts = self.reg.predict(np.array(embeds_ids), np.array(embeds), embeds_text)
        indices = np.argpartition(np.array(predicts), evict_size - 1)[:evict_size]
        selected_embeds_ids = {embeds_ids[i] for i in indices}
        self.evict_cands.update(selected_embeds_ids)
    
    def get_evict_embed_id(self):
        if len(self.evict_cands) == 0 and self.reg.get_train_counter() > 0:
            self.select_evict_cands()
        if len(self.evict_cands) > 0:
            evict_embed_id = self.evict_cands.pop()
            embed, embed_text = self.items.pop(evict_embed_id)
            return evict_embed_id
        else:
            evict_embed_id, (embed, embed_text) = self.items.popitem(last=False)
            return evict_embed_id

    
    def request(self, embeds, embeds_ids, count_nn=1, texts=[]):
        closest_dists, cache_hits_ids = self.get_closest_stored_embeds(embeds, count_nn)
        self.reg.record_for_training(embeds_ids, embeds, texts)
        
        flat_cache_hits_ids = cache_hits_ids.ravel()[cache_hits_ids.ravel() != -1]
        for hit_embed_id in flat_cache_hits_ids:
            self.evict_cands.discard(hit_embed_id)
            self.items.move_to_end(hit_embed_id)
        
        cache_hits = np.sum(closest_dists < self.same_embed_distance, axis=1)
        evicted_items = []
        rejected_items = []
        additions = []
        
        for embed_id, embed, embed_text in zip(embeds_ids, embeds, texts): 
            if self.capacity <= self.size():
                evict_embed_id = self.get_evict_embed_id()
                evicted_items.append(evict_embed_id)
            if self.capacity >= self.size():
                self.items[embed_id] = (embed, embed_text)
                additions.append((embed_id, embed))
            self.curr_embed_id += 1

        if additions:
            additions_embeds = [v for (_, v) in additions]
            additions_ids = [v for (v, _) in additions]
            self.index.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        if evicted_items:
            self.index.remove_ids(np.array(evicted_items))
        return cache_hits, evicted_items + rejected_items


class FreqOPT(Cache):

    def __init__(self, same_embed_distance, deltas_count=4, train_capacity_ratio=1.0, belady_boundary_coe=2.0, dim=384):
        super().__init__(same_embed_distance)
        self.dim = dim
        self.belady_boundary_coe = belady_boundary_coe
        self.deltas_count = deltas_count
        self.train_capacity_ratio = train_capacity_ratio 

    def initialize(self, capacity: int, index):
        self.train_counter = 0
        self.items = []
        self.labeled_count = 0
        self.belady_boundary = np.inf
        self.curr_embed_id = 0
        train_capacity = int(self.train_capacity_ratio * capacity)
        self.reg = FreqReg(train_capacity, train_capacity, self.deltas_count, self.same_embed_distance, self.dim)
        self.training_data = {}
        self.index_train = faiss.IndexIDMap2(faiss.IndexFlatL2(self.dim))
        super().initialize(capacity, index)    
    
    def predict(self, embeds_ids, embeds):
        return self.reg.predict(np.array(embeds_ids), np.array(embeds))
    
    def request(self, embeds, embeds_ids, count_nn=1, texts=[]):
        closest_dists, _ = self.get_closest_stored_embeds(embeds, count_nn)
        self.reg.record_for_training(embeds_ids, embeds)
        cache_hits = np.sum(closest_dists < self.same_embed_distance, axis=1)
        evicted_items = []
        rejected_items = []
        additions = []
        
        if self.reg.is_trained():
            if self.reg.get_train_counter() < self.train_counter:
                self.train_counter += 1
                _, all_embeds_ids, all_embeds = zip(*self.items)
                all_next_hits = self.predict(all_embeds_ids, all_embeds)
                self.items = sorted(list(zip(all_next_hits, -np.array(all_embeds_ids), all_embeds)))
            next_hits = self.predict(embeds_ids, embeds)
            entries = sorted(list(zip(next_hits, -np.array(embeds_ids), embeds)), reverse=True)
        else:
            entries = list(zip(-np.array(embeds_ids), embeds_ids, embeds))
        
        for next_hit, minus_embed_id, embed in entries:
            embed_id = -minus_embed_id            
            if self.capacity <= self.size():
                # remove worst
                max_hit, minus_max_embed_id, max_embed = self.items[-1]
                max_embed_id = -minus_max_embed_id
                if max_hit <= next_hit:
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
