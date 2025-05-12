import faiss
import numpy as np
import lightgbm as lgb
import pandas as pd

def pad_array(arr, N, v):
    current_length = arr.shape[0]
    if current_length < N:
        pad_length = N - current_length
        return np.pad(arr, (pad_length, 0), mode='constant', constant_values=v)
    elif current_length > N:
        return arr[-N:]
    return arr

class FreqReg():
    def __init__(self, train_index_size, examples_per_train, deltas_count = 8, same_embed_distance = 0.5, dim=384):
        self.train_index_size = train_index_size
        self.index_min_embed_id = 0
        self.examples_per_train = examples_per_train
        self.training_data = {}
        self.index_train = faiss.IndexIDMap2(faiss.IndexFlatL2(dim))
        self.reg = None
        self.deltas_count = deltas_count
        self.same_embed_distance = same_embed_distance
        self.dim = dim
        self.train_counter = 0
        
    def get_train_counter(self):
        return self.train_counter
    
    def is_trained(self):
        return self.reg is not None
    
    def train(self):
        print("Training...")
        self.reg = lgb.LGBMRegressor()
        data = pd.DataFrame(self.training_data.values())
        embeds_ids = np.array(list(self.training_data.keys()))
        embeds_ages = embeds_ids.max() - embeds_ids + 1 
        X = np.array(data[0].tolist()).astype(np.float32)
        y = data[1].astype(np.float32).to_list() / embeds_ages
        self.reg.fit(X, y)
        self.training_data = {}
        self.train_counter += 1

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
        if len(self.training_data) >= self.examples_per_train:
            self.train()
        features, cache_hits = self.get_features(embeds_ids, embeds)
        for embed_id, embed_cache_hits in cache_hits.items():
            self.training_data[embed_id] = [features[embed_id], 0]
            for cache_hit_embed_id in embed_cache_hits:
                entry = self.training_data.get(cache_hit_embed_id, None)
                if entry:
                    entry[1] += 1
        count_added = len(embeds)
        if self.index_train.ntotal >= self.train_index_size:
            self.index_train.remove_ids(np.array(list(range(self.index_min_embed_id, self.index_min_embed_id + count_added))))
            self.index_min_embed_id += count_added
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
                embed_id,
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