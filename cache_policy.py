import random
from collections import OrderedDict, defaultdict, deque

class CachePolicy:
    def __init__(self, size: int):
        self.size = size
    
    def log_access(self, item):
        pass

class LRU(CachePolicy):
    def __init__(self, size: int):
        super().__init__(size)
        self.items = OrderedDict()

    def log_access(self, item):
        if item in self.items:
            self.items.move_to_end(item)  # Mark as recently used
        self.items[item] = None  # Value is irrelevant, only keys are tracked
        if len(self.items) > self.size:
            removed_item, _ = self.items.popitem(last=False)  # Remove least recently used
            return removed_item
        return None

class RR(CachePolicy):
    def __init__(self, size: int):
        super().__init__(size)
        self.items = set()

    def log_access(self, item):
        self.items.add(item)
        if len(self.items) > self.size:
            removed_item = random.choice(list(self.items))
            self.items.remove(removed_item)
            return removed_item
        return None

class LFU(CachePolicy):
    def __init__(self, size: int):
        super().__init__(size)
        self.items = set()
        self.freq = defaultdict(int)

    def log_access(self, item):
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
            return removed_item
        return None

class FIFO(CachePolicy):
    def __init__(self, size: int):
        super().__init__(size)
        self.items = deque()

    def log_access(self, item):
        if item not in self.items:
            if len(self.items) >= self.size:
                removed_item = self.items.popleft()
                self.items.append(item)
                return removed_item
            self.items.append(item)
        return None

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
