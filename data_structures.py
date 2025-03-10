import heapq
from collections import deque
inf = float("inf")


class MaxHeap:
    def __init__(self, k):
        self.k = k
        self.pop_at = k+1
        self.min_heap = []

    def insert(self, item): #(key, value)
        # Push the new item onto the heap
        heapq.heappush(self.min_heap, item)
        # If the heap exceeds size k, remove the smallest item
        if len(self.min_heap) == self.pop_at:
            heapq.heappop(self.min_heap)

    def top_k(self):
        # Return the top k items sorted by score in descending order
        return self.min_heap


class MinHeap:
    def __init__(self, k):
        self.k = k
        self.pop_at = k+1
        self.max_heap = []

    def insert(self, key, value):
        heapq.heappush(self.max_heap, (-value, key))
        if len(self.max_heap) == self.pop_at:
            heapq.heappop(self.max_heap)

    def top_k(self, reverse=None):
        result = [(key, -value) for value, key in self.max_heap]
        if reverse is not None:
            result.sort(key=lambda x: x[1], reverse=reverse)
        return result
    
    def keys(self):
        return [key for _, key in self.max_heap]
    
    def values(self):
        return [-value for value, _ in self.max_heap]
    
    def __bool__(self):
        return bool(self.max_heap)


class CumsumDeque:
    def __init__(self, size: int):
        self.size = int(size)
        self.items = deque(maxlen=self.size)
        self.cumsum = deque(maxlen=self.size)
    def append(self, item):
        if not (-inf < item < inf):
            return
        total_sum = self.total_sum
        if len(self) == self.size:
            self.items.popleft()
            self.cumsum.popleft()
        self.items.append(item)
        self.cumsum.append(total_sum + item)
    @property
    def total_sum(self):
        return self.cumsum[-1] if self.cumsum else 0
    def __len__(self):
        return len(self.items)
    def __iter__(self):
        return iter(self.items)
    def __getitem__(self, i):
        return self.items[i]
    def __bool__(self):
        return bool(self.items)
    def __str__(self):
        return str(self.items)
    def __repr__(self):
        return repr(self.items)
    def sum(self, last: int = None):
        if last is None:
            return self.total_sum
        elif isinstance(last, int):
            if 0 <= last < len(self):
                return self.cumsum[-1] - self.cumsum[-last-1]
            elif last == len(self):
                return self.cumsum[-1] - self.cumsum[0] + self.items[0]
            else:
                raise ValueError(f"last must be an integer between 0 and len(deque)={len(self)}")
        else:
            raise ValueError(f"last must be an integer, got {type(last)}")
    def mean(self, last: int = None):
        return self.sum(last) / max(1, (len(self) if last is None else last))
