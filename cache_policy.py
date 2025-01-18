import random
from collections import OrderedDict, defaultdict, deque
from typing import Tuple

import numpy as np
from count_min import CountMin
class CachePolicy:
    def __init__(self):
        self.size = None
    
    def set_size(self, size: int) -> None:
        self.size = size
    
    def log_access(self, item: int, position, distances) -> Tuple[int, bool]:
        return -1

    def count_items(self):
        return len(self.items)
    
class LRU(CachePolicy):
    def __init__(self):
        super().__init__()

    def set_size(self, size: int) -> None:
        self.size = size
        self.items = OrderedDict()
            
    def log_access(self, item: int, position, distances) -> int:
        if item in self.items:
            self.items.move_to_end(item)  # Mark as recently used
        self.items[item] = None  # Value is irrelevant, only keys are tracked
        if len(self.items) > self.size:
            removed_item, _ = self.items.popitem(last=False)  # Remove least recently used
            return removed_item, True
        return -1, True

class RR(CachePolicy):
    def __init__(self):
        super().__init__()
        self.items = set()

    def set_size(self, size: int) -> None:
        self.items = set()
        self.size = size
            
    def log_access(self, item: int, position, distances) -> int:
        self.items.add(item)
        if len(self.items) > self.size:
            removed_item = random.choice(list(self.items))
            self.items.remove(removed_item)
            return removed_item, True
        return -1, True

class LFU(CachePolicy):
    def __init__(self):
        super().__init__()

    def set_size(self, size: int) -> None:
        self.size = size
        self.items = set()
        self.freq = defaultdict(int)
            
    def log_access(self, item: int, position, distances) -> int:
        if item in self.items:
            self.freq[item] += 1
        else:
            self.items.add(item)
            self.freq[item] = 1

        if len(self.items) > self.size:
            # Find the least frequently used item
            min_freq = min(self.freq.values())
            min_freq_items = [i for i in self.items if self.freq[i] == min_freq]
            removed_item = min_freq_items[0]  # Pick the first LFU item
            self.items.remove(removed_item)
            del self.freq[removed_item]
            return removed_item, True
        return -1, True

class FIFO(CachePolicy):
    def __init__(self):
        super().__init__()

    def set_size(self, size: int) -> None:
        self.size = size
        self.items = deque()
            
    def log_access(self, item: int, position, distances) -> int:
        if item not in self.items:
            if len(self.items) >= self.size:
                removed_item = self.items.popleft()
                self.items.append(item)
                return removed_item
            self.items.append(item)
        return -1, True

from scipy.spatial import distance_matrix

class OPT(CachePolicy):
    def __init__(self, embeds, same_embed_distance):
        super().__init__()
        distances = distance_matrix(embeds, embeds)
        n = embeds.shape[0]
        self.embeds_next_hit = np.full(n, -1, dtype=int)
        for i in range(n):
            row_slice = distances[i, i+1:]
            indices = np.where(row_slice < same_embed_distance)[0]
            if indices.size > 0:
                first_match = indices[0]
                j = i + 1 + first_match
                self.embeds_next_hit[i] = j
            else:
                self.embeds_next_hit[i] = -1

    def set_size(self, size: int) -> None:
        self.items = {}
        self.size = size
        self.embed_index = 0
            
    def log_access(self, item: int, position, distances) -> int:
        item_next_hit = self.embeds_next_hit[self.embed_index]
        self.embed_index += 1
        if len(self.items) < self.size or item in self.items:
            self.items[item] = item_next_hit
            return -1, True
        removed_item = min(self.items, key=self.items.get)
        removed_item_next_hit = self.items[removed_item]
        if removed_item_next_hit == -1 or removed_item_next_hit > item_next_hit:
            self.items.pop(removed_item)
            self.items[item] = item_next_hit
            return removed_item, True
        return -1, False

class DensityBased(CachePolicy):
    def __init__(self, cell_size = 0.707 * 1/384):
        super().__init__()
        self.cell_size = cell_size

    def set_size(self, size: int) -> None:
        self.size = size
        self.items = dict()
        self.densities = {}
        self.items_counts = {}
        self.cache_hits = defaultdict(int)
        self.cache_misses = defaultdict(int)
            
    def get_key(self, position):
        return tuple(np.round(position / self.cell_size).astype(int))

    def get_remove_candidate(self, position):
        raise Exception('virtual method')
    
    def log_access(self, item: int, position: float, distances) -> int:
        position_rounded = self.get_key(position)
        if item not in self.items:
            if len(self.items) >= self.size:
                removed_item = self.get_remove_candidate(position)
                if removed_item == -1:
                    self.cache_misses[position_rounded] += 1
                    return -1, False
                removed_position_rounded = self.get_key(self.items[removed_item])
                # self.cache_hits[removed_position_rounded] -= self.items_counts[removed_item]
                self.densities.pop(removed_item)
                self.items.pop(removed_item)
                self.items_counts.pop(removed_item)
            else:
                removed_item = -1
            self.items[item] = position
            self.cache_hits[position_rounded] += 1
            self.densities[item] = self.cache_hits[position_rounded]
            self.items_counts[item] = 1
        else:
            removed_item = -1
            self.cache_hits[position_rounded] += 1
            self.densities[item] = self.cache_hits[position_rounded]
            self.items_counts[item] += 1
        return removed_item, True
    
class MinDensity(DensityBased):
    def get_remove_candidate(self, position):
        return min(self.densities, key=self.densities.get)

class MinCounter(DensityBased):
    def get_remove_candidate(self, position):
        return min(self.items_counts, key=self.items_counts.get)

class ProbMinDensity(DensityBased):
    def get_remove_candidate(self, position):
        min_density = min(self.densities.values())
        removed_item = random.choice([item for item, density in self.densities.items() if density == min_density])
        removed_position_rounded = self.get_key(self.items[removed_item])
        updated_density = self.cache_hits[removed_position_rounded]
        self.densities[removed_item] = updated_density
        thresh = 1 / (updated_density + 1)
        if random.random() < thresh:
            return -1
        return removed_item

class ProbMinCounter(DensityBased):
    def get_remove_candidate(self, position):
        removed_item = min(self.items_counts, key=self.items_counts.get)
        min_counter = self.items_counts[removed_item]
        thresh = 1 / (min_counter + 1)
        if random.random() < thresh:
            return -1
        return removed_item

class ProbMisses(DensityBased):
    def get_remove_candidate(self, position):
        position_rounded = self.get_key(position)
        misses = self.cache_misses[position_rounded]
        removed_item = min(self.items_counts, key=self.items_counts.get)
        min_counter = self.items_counts[removed_item]
        div = max(1, min_counter + 1 - misses)
        thresh = 1 / div
        if random.random() < thresh:
            return -1
        return removed_item

class ProximityScore(CachePolicy):
    def __init__(self, neigh_distance: float, decay: float):
        super().__init__()
        self.neigh_distance = neigh_distance
        self.decay = decay

    def set_size(self, size: int) -> None:
        self.size = size
        self.items = dict()
            
    def get_proximity_score(self, distances):
        sum_distance = sum(d for d in distances if d < self.neigh_distance)
        if sum_distance == 0:
            return 0
        return 1 / sum_distance
    
    def log_access(self, item: int, position, distances) -> int:
        if item in self.items:
            self.items[item] = self.decay * self.items[item] + (1 - self.decay) * self.get_proximity_score(distances)
            return -1, True
        else:
            min_item = -1
            if self.count_items() >= self.size:
                min_item = min(self.items, key=self.items.get)
                self.items.pop(min_item)
            self.items[item] = self.get_proximity_score(distances)
            return min_item, True
            
class TinyLFU(CachePolicy):
    """
    A simple version of TinyLFU that uses:
    1) A CountMinSketch to approximate frequency.
    2) A single LRU data structure for actual caching.
    3) A frequency-based admission policy.
    
    Pseudocode:
    - On every access, update the sketch for 'item'.
    - If 'item' is already cached:
        * Move it to the 'most-recently-used' position in LRU.
    - Else:
        * If the cache is not full, admit 'item' directly.
        * If the cache is full:
            1) Select the LRU victim from the cache (or randomly).
            2) Compare frequency of 'item' with frequency of victim.
               If freq(item) >= freq(victim), evict victim and insert 'item'.
               Otherwise, do not admit 'item' (cache unchanged).
    """
    def __init__(self, size: int, sketch_width=1024, sketch_depth=4):
        super().__init__(size)
        # This OrderedDict will serve as our LRU (the end is MRU).
        # Key: item, Value: unused or metadata (can store frequency, but optional).
        self.items = OrderedDict()
        
        # CountMin sketch for frequency approximation
        # You can adjust width/depth based on error/epsilon requirements
        self.sketch = CountMin(width=sketch_width, depth=sketch_depth)
    
    def log_access(self, item: int, position, distances) -> Tuple[int, bool]:
        # 1) Update frequency sketch
        self.sketch.update(item, 1)

        # 2) If item is already in the cache (LRU), move it to MRU
        if item in self.items:
            # Move to the most-recently-used end
            self.items.move_to_end(item)
            return -1, True  # No eviction, successful access

        # 3) Item is not in cache
        if len(self.items) < self.size:
            # There's space to admit item
            self.items[item] = True
            return -1, True
        else:
            # Cache is full: choose a victim from the LRU
            # Victim = the least-recently-used item (front of OrderedDict)
            victim, _ = self.items.popitem(last=False)
            
            # Compare frequencies: if new item has freq >= victim, admit new item
            freq_item = self.sketch.query(item)
            freq_victim = self.sketch.query(victim)
            if freq_item >= freq_victim:
                # Evict victim and admit new item
                self.items[item] = True
                return victim, True
            else:
                # Re-insert the victim (since we decided not to evict it)
                # Because in "real" TinyLFU you'd only remove the victim from LRU if new item is admitted
                self.items[victim] = True
                return -1, False  # no eviction, new item not admitted

# Example usage
if __name__ == "__main__":
    lru = LRU(size=2)
    print(lru.log_access(1))  # None
    print(lru.log_access(2))  # None
    print(lru.log_access(3))  # 1

    rr = RR(size=2)
    print(rr.log_access(1))  # None
    print(rr.log_access(2))  # None
    print(rr.log_access(3))  # Randomly removes one item

    lfu = LFU(size=2)
    print(lfu.log_access(1))  # None
    print(lfu.log_access(2))  # None
    print(lfu.log_access(1))  # None
    print(lfu.log_access(3))  # 2

    fifo = FIFO(size=2)
    print(fifo.log_access(1))  # None
    print(fifo.log_access(2))  # None
    print(fifo.log_access(3))  # 1
