# Let's implement an improved NaiveVectorStore with bug fixes and capacity growth,
# plus a thorough test suite and a small stress/perf check for update-heavy usage.

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import time

class NaiveVectorStore:
    """
    Minimal in-memory store (L2 only).
    API:
      - add_with_ids(x: float32[n, d], ids: int64[n])
      - remove_ids(ids: int64[m]) -> int
      - search(xq: float32[q, d], k: int) -> (D: float32[q, k], I: int64[q, k])
      - range_search(xq: float32[q, d], radius: float, limit: Optional[int]) -> (List[float32[<=n]], List[int64[<=n]])
    Notes:
      * Distances are **squared L2** (same convention as FAISS).
      * Designed for very update-heavy (add/remove) workloads: uses capacity growth and swap-pop deletes.
    """

    def __init__(self, dim: int = 384, initial_capacity: int = 0):
        assert isinstance(dim, int) and dim > 0
        assert isinstance(initial_capacity, int) and initial_capacity >= 0
        self.dim = dim
        self._cap = int(initial_capacity)
        self._size = 0

        if self._cap == 0:
            self._vecs  = np.empty((0, dim), dtype=np.float32)
            self._ids   = np.empty((0,), dtype=np.int64)
            self._norm2 = np.empty((0,), dtype=np.float32)
        else:
            self._vecs  = np.empty((self._cap, dim), dtype=np.float32)
            self._ids   = np.empty((self._cap,), dtype=np.int64)
            self._norm2 = np.empty((self._cap,), dtype=np.float32)
        self._id2row: dict[int, int] = {}

    # --------------- internal helpers ---------------
    def _ensure_capacity(self, need: int) -> None:
        if self._size + need <= self._cap:
            return
        new_cap = max(1, self._cap)
        while new_cap < self._size + need:
            new_cap = max(new_cap * 2, 4)
        # grow
        if self._cap == 0:
            self._vecs  = np.empty((new_cap, self.dim), dtype=np.float32)
            self._ids   = np.empty((new_cap,), dtype=np.int64)
            self._norm2 = np.empty((new_cap,), dtype=np.float32)
        else:
            self._vecs  = np.resize(self._vecs,  (new_cap, self.dim))
            self._ids   = np.resize(self._ids,   (new_cap,))
            self._norm2 = np.resize(self._norm2, (new_cap,))
        self._cap = new_cap

    def __len__(self) -> int:
        return self._size

    # ----------------------- Public API -----------------------
    def add_with_ids(self, x: np.ndarray, ids: np.ndarray) -> None:
        assert isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[1] == self.dim and x.dtype == np.float32 and x.flags['C_CONTIGUOUS']
        assert isinstance(ids, np.ndarray) and ids.ndim == 1 and ids.dtype == np.int64 and ids.shape[0] == x.shape[0]

        # batch ids uniqueness & no overlaps with existing
        if ids.size > 0:
            # enforce uniqueness inside the batch and against existing
            assert np.unique(ids).size == ids.size
            for i in ids.tolist():
                assert int(i) not in self._id2row

        n_new = x.shape[0]
        if n_new == 0:
            return
        self._ensure_capacity(n_new)

        start = self._size
        end = start + n_new

        # write into the gap
        self._vecs[start:end, :] = x
        self._ids[start:end] = ids
        # squared L2 norm
        self._norm2[start:end] = np.einsum('ij,ij->i', x, x, dtype=np.float32)

        # update map
        for j, vid in enumerate(ids.tolist()):
            self._id2row[int(vid)] = start + j

        self._size = end

    def remove_ids(self, ids: np.ndarray) -> int:
        assert isinstance(ids, np.ndarray) and ids.ndim == 1 and ids.dtype == np.int64
        removed = 0
        for vid in ids.tolist():
            row = self._id2row.get(int(vid))
            if row is None or row >= self._size:
                continue
            last = self._size - 1
            if row != last:
                # swap row <-> last (swap-pop)
                # swap vectors (row-wise)
                tmp = self._vecs[row, :].copy()
                self._vecs[row, :] = self._vecs[last, :]
                self._vecs[last, :] = tmp
                # swap ids and norms
                self._ids[row], self._ids[last] = self._ids[last], self._ids[row]
                self._norm2[row], self._norm2[last] = self._norm2[last], self._norm2[row]
                # fix moved id mapping (id now at 'row')
                moved_id = int(self._ids[row])
                self._id2row[moved_id] = row
            # pop last
            self._size -= 1
            del self._id2row[int(vid)]
            removed += 1
        return removed

    def search(self, xq: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        assert isinstance(xq, np.ndarray) and xq.ndim == 2 and xq.shape[1] == self.dim and xq.dtype == np.float32 and xq.flags['C_CONTIGUOUS']
        assert isinstance(k, int) and k >= 1

        n = self._size
        nq = xq.shape[0]
        if n == 0:
            D = np.full((nq, k), np.inf, dtype=np.float32)
            I = -np.ones((nq, k), dtype=np.int64)
            return D, I

        base = slice(0, n)
        xb = self._vecs[base, :]
        ids = self._ids[base]
        n2 = self._norm2[base]

        qn2 = np.einsum('ij,ij->i', xq, xq, dtype=np.float32)[:, None]  # (nq,1)
        dot = xq @ xb.T                                                 # (nq,n)
        dist2 = qn2 + n2[None, :] - 2.0 * dot                           # (nq,n)

        # numerical floor to 0 for tiny negatives
        np.maximum(dist2, 0.0, out=dist2)

        k_eff = min(k, n)
        # argpartition for partial top-k
        idx_part = np.argpartition(dist2, k_eff - 1, axis=1)[:, :k_eff]        # (nq,k_eff)
        part_vals = np.take_along_axis(dist2, idx_part, axis=1)                # (nq,k_eff)
        order = np.argsort(part_vals, axis=1)                                   # (nq,k_eff)
        top_idx = np.take_along_axis(idx_part, order, axis=1)                   # (nq,k_eff)
        top_vals = np.take_along_axis(dist2, top_idx, axis=1)                   # (nq,k_eff)

        I = ids[top_idx]
        D = top_vals.astype(np.float32, copy=False)

        if k_eff < k:
            pad = k - k_eff
            I = np.hstack([I, -np.ones((nq, pad), dtype=np.int64)])
            D = np.hstack([D, np.full((nq, pad), np.inf, dtype=np.float32)])
        return D, I

    def range_search(
        self,
        xq: np.ndarray,
        radius: float,
        limit: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        assert isinstance(xq, np.ndarray) and xq.ndim == 2 and xq.shape[1] == self.dim and xq.dtype == np.float32 and xq.flags['C_CONTIGUOUS']
        assert isinstance(radius, (float, np.floating)) and radius >= 0.0
        n = self._size
        nq = xq.shape[0]
        cap = int(limit) if limit is not None else n

        if n == 0:
            return [np.empty((0,), dtype=np.float32) for _ in range(nq)], \
                   [np.empty((0,), dtype=np.int64)   for _ in range(nq)]

        base = slice(0, n)
        xb = self._vecs[base, :]
        ids = self._ids[base]
        n2 = self._norm2[base]

        qn2 = np.einsum('ij,ij->i', xq, xq, dtype=np.float32)[:, None]
        dot = xq @ xb.T
        dist2 = qn2 + n2[None, :] - 2.0 * dot
        np.maximum(dist2, 0.0, out=dist2)

        dlists: List[np.ndarray] = []
        ilists: List[np.ndarray] = []
        for i in range(nq):
            idxs = np.flatnonzero(dist2[i] <= radius)
            if idxs.size == 0:
                dlists.append(np.empty((0,), dtype=np.float32))
                ilists.append(np.empty((0,), dtype=np.int64))
                continue
            order = np.argsort(dist2[i, idxs])
            if cap is not None:
                order = order[:cap]
            dlists.append(dist2[i, idxs[order]].astype(np.float32, copy=False))
            ilists.append(ids[idxs[order]].astype(np.int64, copy=False))
        return dlists, ilists

    # Convenience: check internal consistency (used in tests)
    def _check_invariants(self) -> None:
        assert 0 <= self._size <= self._cap
        assert self._vecs.shape == (self._cap, self.dim) or self._vecs.shape == (0, self.dim)
        assert self._ids.shape == (self._cap,) or self._ids.shape == (0,)
        assert self._norm2.shape == (self._cap,) or self._norm2.shape == (0,)
        # id2row matches ids[0:size]
        seen = set()
        for r in range(self._size):
            vid = int(self._ids[r])
            assert self._id2row[vid] == r
            seen.add(vid)
        assert set(self._id2row.keys()) == seen


# ----------------------- Tests -----------------------

def test_add_and_search_basic():
    dim = 8
    store = NaiveVectorStore(dim, initial_capacity=2)
    xb = np.random.random((5, dim)).astype(np.float32, order="C")
    ids = np.arange(10, 15, dtype=np.int64)
    store.add_with_ids(xb, ids)
    assert len(store) == 5

    # exact brute-force distances to check correctness
    xq = np.random.random((3, dim)).astype(np.float32, order="C")
    D, I = store.search(xq, k=5)

    # validate shape & types
    assert D.shape == (3,5) and I.shape == (3,5)
    assert D.dtype == np.float32 and I.dtype == np.int64

    # recompute brute force to verify
    xb2 = xb
    n2 = (xb2*xb2).sum(axis=1, dtype=np.float32)
    qn2 = (xq*xq).sum(axis=1, dtype=np.float32)[:, None]
    dist2 = qn2 + n2[None, :] - 2.0 * (xq @ xb2.T)
    dist2 = np.maximum(dist2, 0.0)
    # check first neighbor matches brute-force top1
    bf_top1 = np.argmin(dist2, axis=1)
    assert np.all(I[:,0] == ids[bf_top1])

def test_remove_ids_and_mappings():
    dim = 4
    store = NaiveVectorStore(dim, initial_capacity=1)
    xb = np.random.random((6, dim)).astype(np.float32, order="C")
    ids = np.array([101,102,103,104,105,106], dtype=np.int64)
    store.add_with_ids(xb, ids)
    store._check_invariants()

    # remove middle, then first, then last
    removed = store.remove_ids(np.array([103,101,106], dtype=np.int64))
    assert removed == 3
    assert len(store) == 3
    store._check_invariants()

    # remaining should be {102,104,105}
    rem = set(store._ids[:store._size].tolist())
    assert rem == {102,104,105}

def test_k_greater_than_n_and_empty():
    dim = 6
    store = NaiveVectorStore(dim)
    # empty search
    xq = np.random.random((2, dim)).astype(np.float32, order="C")
    D, I = store.search(xq, k=5)
    assert np.isinf(D).all() and (I == -1).all()

    # after add, k > n
    xb = np.random.random((3, dim)).astype(np.float32, order="C")
    ids = np.array([1,2,3], dtype=np.int64)
    store.add_with_ids(xb, ids)
    D, I = store.search(xq, k=10)
    assert D.shape == (2,10) and I.shape == (2,10)
    # last columns are pads
    assert np.isinf(D[:,3:]).all() and (I[:,3:] == -1).all()

def test_range_search_and_limit():
    dim = 5
    store = NaiveVectorStore(dim)
    xb = np.array([[0,0,0,0,0],[1,0,0,0,0],[2,0,0,0,0],[3,0,0,0,0]], dtype=np.float32, order="C")
    ids = np.array([10,11,12,13], dtype=np.int64)
    store.add_with_ids(xb, ids)

    xq = np.array([[0.5,0,0,0,0]], dtype=np.float32, order="C")
    # squared distances to points: 0.25, 0.25, 2.25, 6.25
    dl, il = store.range_search(xq, radius=2.25)  # include up to id=12
    assert len(dl) == 1 and len(il) == 1
    assert np.all(il[0] == np.array([10,11,12], dtype=np.int64))
    assert np.allclose(dl[0], np.array([0.25,0.25,2.25], dtype=np.float32))

    # with limit=2
    dl2, il2 = store.range_search(xq, radius=2.25, limit=2)
    assert np.all(il2[0] == np.array([10,11], dtype=np.int64))
    assert np.allclose(dl2[0], np.array([0.25,0.25], dtype=np.float32))

def test_update_heavy_stress():
    dim = 16
    rng = np.random.default_rng(42)
    store = NaiveVectorStore(dim, initial_capacity=1)
    next_id = 1

    # 2000 ops: random add/remove cycles
    live = set()
    for _ in range(2000):
        if len(live) == 0 or rng.random() < 0.6:
            # add batch of 1..4
            bsz = rng.integers(1, 5)
            xb = rng.random((bsz, dim), dtype=np.float32)
            ids = np.arange(next_id, next_id+bsz, dtype=np.int64)
            next_id += bsz
            store.add_with_ids(xb, ids)
            live.update(ids.tolist())
        else:
            # remove batch of up to 3 from live
            if len(live) > 0:
                rm = min(len(live), int(rng.integers(1,4)))
                ids = np.array(list(rng.choice(list(live), size=rm, replace=False)), dtype=np.int64)
                store.remove_ids(ids)
                for i in ids.tolist():
                    live.discard(int(i))
        # check invariants sometimes
        if rng.random() < 0.05:
            store._check_invariants()

    store._check_invariants()
    # final consistency between id set and internal ids
    internal_ids = set(store._ids[:store._size].tolist())
    assert internal_ids == set(int(x) for x in live)

def run_all_tests():
    tests = [
        test_add_and_search_basic,
        test_remove_ids_and_mappings,
        test_k_greater_than_n_and_empty,
        test_range_search_and_limit,
        test_update_heavy_stress,
    ]
    t0 = time.time()
    for t in tests:
        t()
    dt = time.time() - t0
    return f"All tests passed in {dt:.3f}s"

if __name__ == "__main__":
    print(run_all_tests())
