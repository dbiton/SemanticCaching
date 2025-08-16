# faiss_like_hnsw.py
from __future__ import annotations
import hnswlib
import numpy as np
import os
import pickle
from typing import Iterable, Optional, Tuple


class FaissLikeHNSW:
    """
    A thin compatibility layer that makes `hnswlib.Index` feel like a Faiss `Index`.

    Key features:
      - add / add_with_ids (maps to hnswlib.add_items)
      - search (returns (D, I) like Faiss: float32 distances, int64 ids, shape (nq, k))
      - remove_ids (lazy: mark_deleted)
      - unremove_ids (unmark_deleted)   # convenience
      - reconstruct(id) if keep_vectors=True
      - range_search(q, radius): exact CPU fallback if keep_vectors=True; otherwise ValueError
      - reset()
      - ntotal (alive count), total_count (including tombstoned)
      - is_trained (always True), train() no-op
      - set_ef / get_ef
      - save/load via pickle (preserves vectors if keep_vectors=True)

    Notes:
      - hnswlib uses lazy deletion; high deleted ratio may hurt speed/recall.
      - Rebuilding/compaction isn’t automatic; you can instantiate a fresh wrapper
        and re-add only alive ids using self.ids() and reconstruct().
      - `range_search` is an exact (brute-force) fallback if keep_vectors=True.
        It computes distances to all alive vectors with NumPy and filters by radius.
    """

    def __init__(
        self,
        dim: int,
        space: str = "l2",         # 'l2' or 'cosine' (hnswlib naming)
        max_elements: int = 200_000,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        keep_vectors: bool = True,
        dtype: np.dtype = np.float32,
        start_id: int = 0,         # used if you call add() without ids (like Faiss)
    ):
        if space not in ("l2", "cosine", "ip"):
            # hnswlib supports 'l2', 'ip' (inner product); 'cosine' is implemented as normalized IP internally
            # but the Python API accepts 'cosine' too on recent versions. Falling back to 'ip' if asked.
            pass

        self.dim = int(dim)
        self.space = space
        self.max_elements = int(max_elements)
        self.M = int(M)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self.keep_vectors = bool(keep_vectors)
        self.dtype = dtype

        self._index = hnswlib.Index(space=self.space, dim=self.dim)
        self._index.init_index(max_elements=self.max_elements, ef_construction=self.ef_construction, M=self.M)
        self._index.set_ef(self.ef_search)

        # bookkeeping
        self._next_id = int(start_id)
        self._alive = set()          # ids not marked deleted
        self._all_ids = set()        # all ids ever inserted (incl. deleted)
        self._vecs = {} if self.keep_vectors else None  # id -> np.ndarray
        self._is_closed = False

    # ---------- Properties / basic info ----------

    @property
    def is_trained(self) -> bool:
        # HNSW doesn't need training
        return True

    def train(self, x: np.ndarray) -> None:
        # no-op for compatibility
        return

    @property
    def ntotal(self) -> int:
        """Alive (not-deleted) count, like Faiss's ntotal."""
        return len(self._alive)

    @property
    def total_count(self) -> int:
        """Total ever added (alive + tombstoned)."""
        return len(self._all_ids)

    def ids(self) -> np.ndarray:
        """Return alive ids (int64)."""
        return np.array(sorted(self._alive), dtype=np.int64)

    # ---------- EF helpers ----------

    def set_ef(self, ef: int) -> None:
        self.ef_search = int(ef)
        self._index.set_ef(self.ef_search)

    def get_ef(self) -> int:
        return self.ef_search

    # ---------- Add API ----------

    def _ensure_2d(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=self.dtype)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {x.shape[1]}")
        return x

    def add(self, x: np.ndarray) -> np.ndarray:
        """
        Faiss-like add(x): assigns contiguous integer ids starting at _next_id.
        Returns the ids used.
        """
        x = self._ensure_2d(x)
        n = x.shape[0]
        ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
        self._next_id += n
        self.add_with_ids(x, ids)
        return ids

    def add_with_ids(self, x: np.ndarray, ids: np.ndarray, *, replace_deleted: bool = True) -> None:
        """
        Faiss-like add_with_ids(x, ids). When replace_deleted=True, behaves like hnswlib's
        'replace_deleted' to reuse tombstoned slots.
        """
        x = self._ensure_2d(x)
        ids = np.asarray(ids, dtype=np.int64)
        if x.shape[0] != ids.shape[0]:
            raise ValueError("x and ids must have the same number of rows.")
        if np.unique(ids).shape[0] != ids.shape[0]:
            raise ValueError("Duplicate ids in add_with_ids are not allowed.")

        # (Optional) maintain vector store for reconstruct/range_search
        if self.keep_vectors:
            for vid, vec in zip(ids.tolist(), x):
                self._vecs[int(vid)] = np.ascontiguousarray(vec, dtype=self.dtype)

        # hnswlib add
        self._index.add_items(x, ids, replace_deleted=replace_deleted)

        # bookkeeping
        for vid in ids.tolist():
            self._all_ids.add(int(vid))
            self._alive.add(int(vid))

    # ---------- Search API ----------

    def search(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Faiss-like search: returns (D, I) with shapes (nq, k):
          - D: float32 distances
          - I: int64 ids  (dead slots filled with -1 if fewer than k exist)
        """
        if k <= 0:
            raise ValueError("k must be positive.")
        q = self._ensure_2d(q)
        labels, dists = self._index.knn_query(q, k=k)  # returns labels(int64), distances(float32)
        # hnswlib may return -1s when fewer than k results are available (e.g., many deletions)
        D = np.asarray(dists, dtype=np.float32, order="C")
        I = np.asarray(labels, dtype=np.int64, order="C")
        return (D, I)

    def range_search(self, q: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Faiss-style range_search(q, radius) with an exact fallback:
          Returns (lims, D, I) where:
            - lims shape (nq+1,), lims[i]..lims[i+1]-1 are matches for query i
            - D, I are 1-D arrays of float32 distances and int64 ids, concatenated.
        Requires keep_vectors=True (so we can do exact CPU distances). If not set, raises ValueError.

        Complexity: O(nq * alive_count * dim). Use sparingly or batch small queries.
        """
        if not self.keep_vectors:
            raise ValueError("range_search requires keep_vectors=True to do exact fallback.")

        q = self._ensure_2d(q)
        alive_ids = np.fromiter(self._alive, dtype=np.int64)
        if alive_ids.size == 0:
            # empty
            lims = np.zeros(q.shape[0] + 1, dtype=np.int64)
            return lims, np.empty(0, np.float32), np.empty(0, np.int64)

        # Stack alive vectors
        X = np.stack([self._vecs[int(i)] for i in alive_ids]).astype(self.dtype, copy=False)  # (N, d)

        # Exact distances: L2 or cosine/IP depending on space
        if self.space == "l2":
            # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
            q2 = (q ** 2).sum(axis=1, keepdims=True)      # (nq,1)
            x2 = (X ** 2).sum(axis=1, keepdims=True).T    # (1,N)
            G = q @ X.T                                   # (nq,N)
            Dfull = q2 + x2 - 2.0 * G
            Dfull = np.maximum(Dfull, 0.0, out=Dfull)    # numerical safety
        elif self.space in ("ip", "cosine"):
            # if cosine, assume vectors are normalized; distance = 1 - cosine_sim
            # Do inner product first:
            G = q @ X.T                                   # (nq,N)
            if self.space == "ip":
                # For "IP", a “distance” equivalent is -G if you want to sort ascending.
                # We'll map to a distance-like metric: D = -G
                Dfull = -G
            else:
                # cosine distance
                # normalize if not already (best if inputs were normalized at add time)
                qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
                xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
                G = qn @ xn.T
                Dfull = 1.0 - G
        else:
            raise ValueError(f"Unsupported space for range_search: {self.space}")

        # filter by radius
        nq = q.shape[0]
        lims = np.zeros(nq + 1, dtype=np.int64)
        Ds: list[np.ndarray] = []
        Is: list[np.ndarray] = []
        for i in range(nq):
            mask = Dfull[i] <= radius
            d_i = Dfull[i, mask].astype(np.float32, copy=False)
            i_i = alive_ids[mask].astype(np.int64, copy=False)
            Ds.append(d_i)
            Is.append(i_i)
            lims[i + 1] = lims[i] + d_i.shape[0]

        if lims[-1] == 0:
            return lims, np.empty(0, np.float32), np.empty(0, np.int64)

        D_concat = np.concatenate(Ds)
        I_concat = np.concatenate(Is)
        return lims, D_concat, I_concat

    # ---------- Deletions ----------

    def remove_ids(self, ids: Iterable[int]) -> int:
        """
        Faiss-like remove_ids: performs lazy deletion (tombstones).
        Returns number of ids successfully marked deleted.
        """
        ids = np.asarray(list(ids), dtype=np.int64)
        count = 0
        for vid in ids.tolist():
            if vid in self._alive:
                self._index.mark_deleted(int(vid))
                self._alive.discard(int(vid))
                count += 1
        return count

    def unremove_ids(self, ids: Iterable[int]) -> int:
        """Convenience: unmark tombstoned ids. Returns count successfully restored."""
        ids = np.asarray(list(ids), dtype=np.int64)
        count = 0
        for vid in ids.tolist():
            if vid in self._all_ids and vid not in self._alive:
                self._index.unmark_deleted(int(vid))
                self._alive.add(int(vid))
                count += 1
        return count

    # ---------- Reconstruct / getters ----------

    def reconstruct(self, vid: int) -> np.ndarray:
        """
        Return the stored vector for id `vid`.
        Requires keep_vectors=True and the id must exist (even if currently deleted).
        """
        if not self.keep_vectors:
            raise ValueError("reconstruct requires keep_vectors=True.")
        try:
            return self._vecs[int(vid)].copy()
        except Exception as e:
            raise KeyError(f"id {vid} not found in stored vectors") from e

    # ---------- Maintenance ----------

    def reset(self) -> None:
        """Clear everything and reinitialize the index."""
        self.__init__(
            dim=self.dim,
            space=self.space,
            max_elements=self.max_elements,
            M=self.M,
            ef_construction=self.ef_construction,
            ef_search=self.ef_search,
            keep_vectors=self.keep_vectors,
            dtype=self.dtype,
            start_id=self._next_id,   # preserve id monotonicity if you like; or set to 0
        )

    # ---------- Save / Load ----------

    def save(self, path: str) -> None:
        """
        Save wrapper (index + metadata) to a directory.
        This uses pickle for the Python state and hnswlib's native save for the graph.
        """
        os.makedirs(path, exist_ok=True)
        # Save hnsw graph
        index_path = os.path.join(path, "hnsw.index")
        self._index.save_index(index_path)
        # Save python-side state
        state = {
            "dim": self.dim,
            "space": self.space,
            "max_elements": self.max_elements,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "keep_vectors": self.keep_vectors,
            "dtype": str(self.dtype.name),
            "_next_id": self._next_id,
            "_alive": list(self._alive),
            "_all_ids": list(self._all_ids),
            "_vecs": self._vecs if self.keep_vectors else None,
        }
        with open(os.path.join(path, "state.pkl"), "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "FaissLikeHNSW":
        """
        Load wrapper saved by save().
        """
        with open(os.path.join(path, "state.pkl"), "rb") as f:
            state = pickle.load(f)

        obj = cls(
            dim=state["dim"],
            space=state["space"],
            max_elements=state["max_elements"],
            M=state["M"],
            ef_construction=state["ef_construction"],
            ef_search=state["ef_search"],
            keep_vectors=state["keep_vectors"],
            dtype=np.dtype(state["dtype"]),
            start_id=state["_next_id"],
        )
        # Reload hnsw graph
        obj._index.load_index(os.path.join(path, "hnsw.index"))
        obj._index.set_ef(obj.ef_search)

        # Restore bookkeeping
        obj._alive = set(map(int, state["_alive"]))
        obj._all_ids = set(map(int, state["_all_ids"]))
        obj._vecs = state["_vecs"]
        return obj

    # ---------- Context manager ----------

    def close(self) -> None:
        """No real resources to release, but provided for API symmetry."""
        self._is_closed = True

    def __enter__(self) -> "FaissLikeHNSW":
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# ---------------------------
# Minimal usage example
# ---------------------------
if __name__ == "__main__":
    d = 64
    idx = FaissLikeHNSW(dim=d, space="l2", max_elements=10000, keep_vectors=True)
    x = np.random.randn(1000, d).astype(np.float32)
    ids = np.arange(1000, dtype=np.int64)

    idx.add_with_ids(x, ids)
    D, I = idx.search(x[:3], k=5)
    print("search distances:\n", D)
    print("search labels:\n", I)

    # lazy delete a few
    removed = idx.remove_ids([10, 11, 12])
    print("removed:", removed, "ntotal:", idx.ntotal)

    # exact range search (requires keep_vectors=True)
    lims, Dr, Ir = idx.range_search(x[:2], radius=50.0)
    print("range lims:", lims, "count:", Dr.shape[0])

    # reconstruct
    v10 = idx.reconstruct(10)
    print("reconstructed 10:", v10[:5])

    # save / load
    idx.save("./tmp_hnsw")
    idx2 = FaissLikeHNSW.load("./tmp_hnsw")
    print("loaded ntotal:", idx2.ntotal)
