import random
from collections import OrderedDict, defaultdict, deque
from typing import Tuple

import numpy as np
from count_min import CountMin
class CachePolicy:
    def __init__(self, size: int):
        self.size = size
    
    def log_access(self, item: int, position) -> Tuple[int, bool]:
        return -1

class LRU(CachePolicy):
    def __init__(self, size: int):
        super().__init__(size)
        self.items = OrderedDict()

    def log_access(self, item: int, position) -> int:
        if item in self.items:
            self.items.move_to_end(item)  # Mark as recently used
        self.items[item] = None  # Value is irrelevant, only keys are tracked
        if len(self.items) > self.size:
            removed_item, _ = self.items.popitem(last=False)  # Remove least recently used
            return removed_item, True
        return -1, True

class RR(CachePolicy):
    def __init__(self, size: int):
        super().__init__(size)
        self.items = set()

    def log_access(self, item: int, position) -> int:
        self.items.add(item)
        if len(self.items) > self.size:
            removed_item = random.choice(list(self.items))
            self.items.remove(removed_item)
            return removed_item, True
        return -1, True

class LFU(CachePolicy):
    def __init__(self, size: int):
        super().__init__(size)
        self.items = set()
        self.freq = defaultdict(int)

    def log_access(self, item: int, position) -> int:
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
    def __init__(self, size: int):
        super().__init__(size)
        self.items = deque()

    def log_access(self, item: int, position) -> int:
        if item not in self.items:
            if len(self.items) >= self.size:
                removed_item = self.items.popleft()
                self.items.append(item)
                return removed_item
            self.items.append(item)
        return -1, True

class DensityBased(CachePolicy):
    def __init__(self, size: int, cell_size = 10 * 1/384):
        super().__init__(size)
        self.items = dict()
        self.densities = {}
        self.items_counts = {}
        self.sketch = CountMin(size, 3)
        self.sketch_misses = CountMin(size, 3)
        self.cell_size = cell_size

    def get_key(self, position):
        return np.round(position / self.cell_size).astype(int)

    def get_remove_candidate(self, position):
        raise Exception('virtual method')
    
    def log_access(self, item: int, position: float) -> int:
        position_rounded = self.get_key(position)
        if item not in self.items:
            if len(self.items) >= self.size:
                removed_item = self.get_remove_candidate(position)
                if removed_item == -1:
                    self.sketch_misses.update_and_query(position_rounded, 1)
                    return -1, False
                removed_position_rounded = self.get_key(self.items[removed_item])
                self.sketch.update_and_query(removed_position_rounded, -self.items_counts[removed_item])
                self.densities.pop(removed_item)
                self.items.pop(removed_item)
                self.items_counts.pop(removed_item)
            else:
                removed_item = -1
            self.items[item] = position
            self.densities[item] = self.sketch.update_and_query(position_rounded, 1)
            self.items_counts[item] = 1
        else:
            removed_item = -1
            self.densities[item] = self.sketch.update_and_query(position_rounded, 1)
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
        updated_density = self.sketch.update_and_query(removed_position_rounded, 0)
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
        misses = self.sketch_misses.update_and_query(position_rounded, 0)
        removed_item = min(self.items_counts, key=self.items_counts.get)
        min_counter = self.items_counts[removed_item]
        div = max(1, min_counter + 1 - misses)
        thresh = 1 / div
        if random.random() < thresh:
            return -1
        return removed_item

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
