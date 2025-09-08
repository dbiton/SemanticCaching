from __future__ import annotations
from typing import Iterable, List, Optional, Tuple, Union, Dict
import numpy as np
import hnswlib
import threading


class HNSWVectorStore:
    """
    hnswlib-backed vector store with a FAISS-like interface:

      - add_with_ids(x, ids)
      - remove_ids(ids)         # marks deleted; returns count
      - search(xq, k)           # -> (D, I) shapes (nq, k)
      - range_search(xq, radius, limit=None)  # variable-length per-query lists

    Notes
    -----
    * No auto-normalization; pass cosine/IP/L2 as you intend.
    * Deletions are lazy (tombstones). Use `rebuild()` to compact.
    * Capacity grows automatically via `ensure_capacity`.
    * Threading: uses hnswlib.set_num_threads() for search/add.

    Parameters
    ----------
    dim : int
        Vector dimensionality.
    space : str
        One of {"l2", "ip", "cosine"}.
    M : int
        HNSW graph connectivity (default 16).
    ef_construction : int
        EF during construction (default 200).
    ef : int
        EF for queries (default 64).
    max_elements : int
        Initial capacity; will auto-grow when needed.
    allow_replace_delete : bool
        If True, hnswlib will re-use deleted slots on add.
    """

    def __init__(
        self,
        dim: int = 384,
        space: str = "l2",
        M: int = 16,
        ef_construction: int = 200,
        ef: int = 64,
        max_elements: int = 100_000,
        allow_replace_delete: bool = False,
        num_threads: Optional[int] = None,
    ):
        # If True, add_items can reuse deleted labels (faster steady-state updates)
        self.allow_replace_delete = allow_replace_delete
        
        self.dim = int(dim)
        self.space = space
        self._index = hnswlib.Index(space=space, dim=dim)
        self._index.init_index(
            max_elements=int(max_elements),
            M=int(M),
            ef_construction=int(ef_construction),
            random_seed=42,
            allow_replace_deleted=self.allow_replace_delete
        )
        self._index.set_ef(int(ef))
        if num_threads is not None:
            self._index.set_num_threads(int(num_threads))

        # We keep an optional id->vector map to support rebuild() and range_search caps.
        # (hnswlib cannot enumerate vectors by id)
        self._vectors: Dict[int, np.ndarray] = {}
        self._deleted: set[int] = set()
        self._lock = threading.RLock()  # guard add/remove/resize/rebuild

        # Track total "known" ids, including deleted ones, so we know capacity needs
        self._live_count = 0

    # ----------------------- Public API (FAISS-like) -----------------------

    def add_with_ids(self, x: np.ndarray, ids: Union[np.ndarray, List[int]]) -> None:
        self.check_rebuild()
        x = self._as_float32_2d(x)
        ids = self._as_int64_1d(ids)
        if len(ids) != len(x):
            raise ValueError(f"ids length {len(ids)} != vectors length {len(x)}")

        with self._lock:
            self.ensure_capacity(self._live_count + len(ids))
            # add_items can accept Python lists or numpy arrays
            self._index.add_items(x, ids, replace_deleted=self.allow_replace_delete)
            for vid, v in zip(ids, x):
                self._vectors[int(vid)] = v
                if int(vid) in self._deleted:
                    self._deleted.remove(int(vid))
                self._live_count += 1

    def remove_ids(self, ids: Union[np.ndarray, List[int]]) -> int:
        self.check_rebuild()
        ids = self._as_int64_1d(ids)
        removed = 0
        with self._lock:
            for vid in ids:
                vid = int(vid)
                try:
                    self._index.mark_deleted(vid)
                    removed += 1
                    self._deleted.add(vid)
                    # keep vector in map for potential rebuild()
                    if vid in self._vectors:
                        del self._vectors[vid]  # optional: free memory immediately
                    self._live_count = max(0, self._live_count - 1)
                except RuntimeError:
                    # hnswlib throws if label not found; ignore for FAISS-like behavior
                    pass
        return removed

    def search(
        self,
        xq: np.ndarray,
        k: int = 10,
        search_params: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (D, I) with shapes (nq, k), like FAISS.
        You can override ef at query-time via search_params={"ef": <int>}.
        """
        xq = self._as_float32_2d(xq)
        if search_params and "ef" in search_params:
            self._index.set_ef(int(search_params["ef"]))

        # hnswlib returns (labels, distances)
        actual_k = min(k, self.ntotal())
        labels, distances = self._index.knn_query(xq, k=int(actual_k))

        # Convert to FAISS order: (D, I)
        I = labels.astype(np.int64, copy=False)
        D = distances.astype(np.float32, copy=False)
        return D, I

    def range_search(
        self,
        xq: np.ndarray,
        radius: float,
        limit: Optional[int] = None,
        search_params: Optional[Dict] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        hnswlib has no native range query; approximate by oversampled KNN + filter.
        Strategy:
          1) choose k' big enough (min(index_size, limit*α)), where α is an oversample.
          2) run knn_query with ef possibly increased (search_params["ef"]).
          3) keep pairs with dist <= radius; cap to `limit` if provided.

        Returns:
          (list_of_dists, list_of_ids), one entry per query.
        """
        xq = self._as_float32_2d(xq)
        if search_params and "ef" in search_params:
            self._index.set_ef(int(search_params["ef"]))

        with self._lock:
            n_elements = self.ntotal()

        # Pick an oversampled k'
        oversample = int(search_params.get("oversample", 5) if search_params else 5)
        cap = int(limit) if limit is not None else 10_000
        kprime = min(n_elements, max(cap * oversample, 32))

        lbls, dists = self._index.knn_query(xq, k=int(kprime))

        ids_list: List[np.ndarray] = []
        dists_list: List[np.ndarray] = []
        for row_ids, row_ds in zip(lbls, dists):
            mask = row_ds <= float(radius)
            ids_f = row_ids[mask]
            dists_f = row_ds[mask]
            if limit is not None and len(ids_f) > cap:
                ids_f = ids_f[:cap]
                dists_f = dists_f[:cap]
            ids_list.append(ids_f.astype(np.int64, copy=False))
            dists_list.append(dists_f.astype(np.float32, copy=False))
        return dists_list, ids_list

    # --------------------------- Maintenance ops ---------------------------

    def check_rebuild(self) -> bool:
        deleted_frac = len(self._deleted) / (1+(self.ntotal() + len(self._deleted)))
        if deleted_frac >= 0.25:
            self.rebuild()
            return True
        return False
        
    def set_ef(self, ef: int) -> None:
        self._index.set_ef(int(ef))

    def set_num_threads(self, n: int) -> None:
        self._index.set_num_threads(int(n))

    def ntotal(self) -> int:
        # hnswlib doesn't expose a direct "live count"; approximate by index.get_current_count()
        try:
            return int(self._index.get_current_count())
        except Exception:
            # Fallback (shouldn't happen): live_count snapshot
            return max(0, self._live_count)

    def ensure_capacity(self, need: int) -> None:
        cur = self._index.get_max_elements()
        if need > cur:
            # grow by 2x until we fit
            new_cap = cur
            while new_cap < need:
                new_cap = max(new_cap * 2, need)
            self._index.resize_index(int(new_cap))

    def rebuild(self, M: Optional[int] = None, ef_construction: Optional[int] = None) -> None:
        """
        Compact the index by re-building from current live vectors.
        Requires `self._vectors` to contain all live vectors; if you've been
        deleting vectors from the map to save memory, you can't rebuild.
        """
        with self._lock:
            if not self._vectors:
                # Nothing to rebuild from
                return
            # Snapshot all live items
            ids = np.fromiter(self._vectors.keys(), dtype=np.int64)
            if len(ids) == 0:
                return
            xb = np.vstack([self._vectors[int(i)] for i in ids]).astype(np.float32, copy=False)

            # Create a fresh index
            M_new = int(M) if M is not None else None
            efc_new = int(ef_construction) if ef_construction is not None else None

            M_use = M_new if M_new is not None else 16
            efc_use = efc_new if efc_new is not None else 200

            new_index = hnswlib.Index(space=self.space, dim=self.dim)
            new_index.init_index(max_elements=max(len(ids), self._index.get_max_elements()),
                                 M=M_use, ef_construction=efc_use, allow_replace_deleted=self.allow_replace_delete)

            new_index.add_items(xb, ids, replace_deleted=self.allow_replace_delete)

            # Swap in
            self._index = new_index
            self._deleted.clear()

    def drop(self) -> None:
        """Release index memory by dropping references."""
        with self._lock:
            self._index = hnswlib.Index(space=self.space, dim=self.dim)  # fresh tiny shell
            self._index.init_index(max_elements=1, M=4, ef_construction=8)
            self._vectors.clear()
            self._deleted.clear()
            self._live_count = 0

    # ----------------------------- Utilities -------------------------------

    @staticmethod
    def _as_float32_2d(x: Union[np.ndarray, Iterable[Iterable[float]]]) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"expected 2D array, got shape {arr.shape}")
        return arr

    @staticmethod
    def _as_int64_1d(ids: Union[np.ndarray, Iterable[int]]) -> np.ndarray:
        if isinstance(ids, np.ndarray):
            arr = ids.astype(np.int64, copy=False)
        else:
            arr = np.asarray(list(ids), dtype=np.int64)
        if arr.ndim != 1:
            raise ValueError("ids must be 1D")
        return arr

if __name__ == "__main__":
    dim = 64
    store = HNSWVectorStore(dim=dim, space="l2", M=16, ef_construction=100, ef=64, max_elements=10_000)

    # Add data
    xb = np.random.random((1000, dim)).astype(np.float32)
    ids = np.arange(1000, dtype=np.int64)
    store.add_with_ids(xb, ids)

    # Search
    xq = np.random.random((5, dim)).astype(np.float32)
    D, I = store.search(xq, k=5, search_params={"ef": 128})
    print("search distances shape:", D.shape, "ids shape:", I.shape)

    # Range search (approximate)
    dists_list, ids_list = store.range_search(xq, radius=2.0, limit=50, search_params={"ef": 200, "oversample": 8})
    print("range results lens:", [len(v) for v in ids_list])

    # Remove
    removed = store.remove_ids([0, 1, 2])
    print("removed:", removed)

    # Optional: compact after many deletes
    store.rebuild()
