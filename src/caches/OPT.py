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

from util.surprisal import calculate_surprisal
import re
from scipy.stats._stats_py import median_abs_deviation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from src.util.reduce_dim import cluster_complete_linkage_faiss, cluster_embeddings_faiss, greedy_cluster_faiss

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
        self.curr_embed_id = max(embeds_ids)
        
        stale_items = [eid for (eid, next_hit) in self.items.items() if next_hit <= self.curr_embed_id]
        if len(stale_items) > 0:
            self.items.update(self.get_next_hits(stale_items))
        
        embeds_next_hits = self.get_next_hits(embeds_ids)
        for embed, embed_id in zip(embeds, embeds_ids):
            embed_next_hit = embeds_next_hits[embed_id] 
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
        embeds_clusters_ids = self.embeds_clusters[embeds_ids]
        curr_id = self.curr_embed_id
        next_hits = {}
        for embed_id, row in zip(embeds_ids, self.embeds_covers[embeds_clusters_ids]):
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
                self.items.pop(evicted_id, None)

            if self.capacity > self.size():
                self.items[embed_id] = embed_next_hit
                additions.append((embed_id, embed))

        if additions:
            additions_embeds = [v for (_, v) in additions]
            additions_ids = [v for (v, _) in additions]
            self.index.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        if evicted_items:
            self.index.remove_ids(np.array(evicted_items))

        return cache_hits, evicted_items + rejected_items

class ClusterOPT(Cache):
    def __init__(self, same_embed_distance, embeds):
        super().__init__(same_embed_distance)
        self.embeds_covers, self.embeds_clusters = self.create_embeds_covers(embeds, same_embed_distance)

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
        super().initialize(capacity, index)
    
    def get_next_hits(self, embeds_ids):
        embeds_clusters_ids = self.embeds_clusters[embeds_ids]
        curr_id = self.curr_embed_id
        next_hits = {}
        for embed_id, row in zip(embeds_ids, self.embeds_covers[embeds_clusters_ids]):
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
        self.curr_embed_id = max(embeds_ids)
        
        stale_items = [eid for (eid, next_hit) in self.items.items() if next_hit <= self.curr_embed_id]
        if len(stale_items) > 0:
            self.items.update(self.get_next_hits(stale_items))
        
        embeds_next_hits = self.get_next_hits(embeds_ids)
        for embed, embed_id in zip(embeds, embeds_ids):
            embed_next_hit = embeds_next_hits[embed_id] 
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
        
    
    def request(self, embeds, embeds_ids, count_nn=1, texts=[]):
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
            self.index.add_with_ids(np.array(additions_embeds), np.array(additions_ids))
        if evicted_items:
            self.index.remove_ids(np.array(evicted_items))
        return cache_hits, evicted_items + rejected_items
