import faiss
import numpy as np
from typing import Iterable, Optional, Callable, Tuple, Dict, Set

class DeletesOnlyWrapper:
    """
    Deletes-only wrapper for FAISS, meant for: IndexIDMap2(IndexHNSWFlat(...)).

    Assumptions:
      - You handle all preprocessing yourself (e.g., normalization for cosine/IP).
      - Metric on the base index is already set correctly (e.g., METRIC_INNER_PRODUCT for cosine).

    Features:
      - Lazy deletions via tombstones.
      - Search that oversamples + filters tombstones ONLY when needed.
      - Optional adaptive ef bump when filtering is in effect.
      - Rebuild that physically drops tombstones.
      - Returns (D, I) exactly like FAISS.
      - Passthrough to raw index.search when no tombstones/filters => exact parity.

    Notes:
      - You MUST add via this wrapper so it can retain id->vector for rebuilds.
    """

    def __init__(
        self,
        index: faiss.Index,                 # expect IndexIDMap2 over IndexHNSWFlat
        *,
        ef_query: Optional[int] = None,     # if given, sets hnsw.efSearch once
        ef_construction: int = 400,         # used only during rebuild
        deleted_rebuild_threshold: float = 0.2,
        oversample_factor: float = 3.0,
        max_oversample: int = 4096,
        adapt_ef_when_filtering: bool = True,
    ):
        self.index = index
        self.dim = index.d
        self.ef_construction = int(ef_construction)
        self.deleted_rebuild_threshold = float(deleted_rebuild_threshold)
        self.oversample_factor = float(oversample_factor)
        self.max_oversample = int(max_oversample)
        self.adapt_ef_when_filtering = bool(adapt_ef_when_filtering)

        self._id_to_vec: Dict[int, np.ndarray] = {}
        self._deleted: Set[int] = set()

        # Observability for rebuilds
        self._M = getattr(getattr(self.index, "hnsw", None), "M", None)
        self._metric_type = getattr(self.index, "metric_type", faiss.METRIC_L2)

        # Optional initial ef
        if ef_query is not None and hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = int(ef_query)

    # -------- utilities --------
    def _is_hnsw(self) -> bool:
        return hasattr(self.index, "hnsw")

    def _live_count(self) -> int:
        return len(self._id_to_vec) - len(self._deleted)

    def _deleted_ratio(self) -> float:
        total = len(self._id_to_vec)
        return 0.0 if total == 0 else len(self._deleted) / total

    # -------- public API --------

    def set_ef(self, ef: int):
        if self._is_hnsw():
            self.index.hnsw.efSearch = int(max(1, ef))

    def add_with_ids(self, X: np.ndarray, ids: np.ndarray, *, upsert: bool = False):
        X = np.asarray(X, dtype=np.float32, order="C")
        ids = np.asarray(ids, dtype=np.int64, order="C")
        assert X.ndim == 2 and X.shape[1] == self.dim and ids.shape[0] == X.shape[0]

        to_add_X, to_add_ids = [], []
        for v, i in zip(X, ids):
            i = int(i)
            if i in self._id_to_vec:
                if not upsert:
                    raise ValueError(f"ID {i} already exists; use upsert=True.")
                # logical replace: tombstone old physical entry; keep new in memory
                self._id_to_vec[i] = v.copy()
                self._deleted.add(i)
                continue
            self._id_to_vec[i] = v.copy()
            to_add_X.append(v)
            to_add_ids.append(i)

        if to_add_ids:
            self.index.add_with_ids(
                np.asarray(to_add_X, dtype=np.float32, order="C"),
                np.asarray(to_add_ids, dtype=np.int64, order="C"),
            )

    def add_items(self, X: np.ndarray, ids: Optional[np.ndarray] = None, *, upsert: bool = False):
        X = np.asarray(X, dtype=np.float32, order="C")
        assert X.ndim == 2 and X.shape[1] == self.dim
        if ids is None:
            start = 0 if not self._id_to_vec else (max(self._id_to_vec.keys()) + 1)
            ids = np.arange(start, start + X.shape[0], dtype=np.int64)
        self.add_with_ids(X, ids, upsert=upsert)

    def mark_deleted(self, ids: Iterable[int]):
        for i in ids:
            i = int(i)
            if i in self._id_to_vec:
                self._deleted.add(i)

    def remove_ids(self, ids: Iterable[int]):
        self.mark_deleted(ids)

    def search(
        self,
        Q: np.ndarray,
        k: int,
        *,
        ef: Optional[int] = None,
        filter_fn: Optional[Callable[[int], bool]] = None,
        pad_with_minus_one: bool = True,
        passthrough_if_no_deletes: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (D, I) like FAISS.
        - If there are NO tombstones and no filter, optionally passthrough to raw index.search.
        - Otherwise, oversample + filter tombstones (and optional filter_fn).
        - Adaptive ef bumping only when filtering is in effect (and enabled).
        """
        Q = np.asarray(Q, dtype=np.float32, order="C")
        assert Q.ndim == 2 and Q.shape[1] == self.dim

        # Optional ef override (no hidden normalization/bumping)
        if ef is not None and self._is_hnsw():
            self.index.hnsw.efSearch = int(max(1, ef))

        live = self._live_count()
        k_eff = min(k, max(0, live))

        # Fast path: exact passthrough for parity
        if passthrough_if_no_deletes and not self._deleted and filter_fn is None:
            return self.index.search(Q, k)

        if k_eff == 0:
            nq = Q.shape[0]
            D = np.full((nq, 0), np.inf, dtype=np.float32)
            I = np.full((nq, 0), -1,  dtype=np.int64)
            return D, I

        # Initial oversampling budget
        k_try = max(k_eff + 16, int(np.ceil(k_eff * self.oversample_factor)))
        k_try = min(self.max_oversample, max(k_try, k_eff))

        nq = Q.shape[0]
        outD = np.full((nq, k_eff), np.inf, dtype=np.float32)
        outI = np.full((nq, k_eff), -1,  dtype=np.int64)

        def keep(label: int) -> bool:
            if label == -1:
                return False
            if label in self._deleted:
                return False
            return filter_fn(label) if filter_fn else True

        while True:
            D, I = self.index.search(Q, k_try)

            all_ok = True
            for qi in range(nq):
                labs = I[qi]
                dsts = D[qi]
                mask = [keep(int(l)) for l in labs]
                if any(mask):
                    labs = labs[mask]
                    dsts = dsts[mask]
                else:
                    labs = np.empty((0,), dtype=labs.dtype)
                    dsts = np.empty((0,), dtype=dsts.dtype)

                m = min(k_eff, labs.shape[0])
                if m < k_eff:
                    all_ok = False
                if m:
                    outI[qi, :m] = labs[:m]
                    outD[qi, :m] = dsts[:m]

            if all_ok:
                return outD, outI  # FAISS order

            # Need more candidates? Optionally bump ef & oversample.
            bumped = False
            if self.adapt_ef_when_filtering and self._is_hnsw():
                cur_ef = self.index.hnsw.efSearch
                next_ef = min(4096, max(cur_ef * 2, 16 * k_eff))
                if next_ef > cur_ef:
                    self.index.hnsw.efSearch = next_ef
                    bumped = True

            next_k_try = min(self.max_oversample, max(k_try * 2, k_eff + 32))
            if next_k_try > k_try:
                k_try = next_k_try
                bumped = True

            if not bumped:
                if pad_with_minus_one:
                    return outD, outI
                raise RuntimeError("Not enough live results to fill k after filtering.")

    def maybe_rebuild(self, *, force: bool = False):
        """Physically drop tombstones by rebuilding a fresh HNSW and re-adding live vectors."""
        if not force and self._deleted_ratio() < self.deleted_rebuild_threshold:
            return

        if self._M is None:
            self._M = getattr(getattr(self.index, "hnsw", None), "M", 32)

        live_ids = [i for i in self._id_to_vec.keys() if i not in self._deleted]

        # Recreate base HNSW with same M and metric
        base = faiss.IndexHNSWFlat(self.dim, int(self._M))
        base.metric_type = self._metric_type
        try:
            base.hnsw.efConstruction = self.ef_construction
        except Exception:
            pass

        new_index = faiss.IndexIDMap2(base)

        if live_ids:
            live_ids_np = np.asarray(live_ids, dtype=np.int64)
            live_vecs = np.vstack([self._id_to_vec[i] for i in live_ids_np]).astype(np.float32, copy=False)
            new_index.add_with_ids(live_vecs, live_ids_np)

        self.index = new_index
        self._deleted.clear()

    # observability
    def live_count(self) -> int:
        return self._live_count()

    def deleted_ratio(self) -> float:
        return self._deleted_ratio()
