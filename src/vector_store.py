from __future__ import annotations
from typing import Iterable, List, Optional, Tuple, Union, Dict
import numpy as np

try:
    from pymilvus import MilvusClient, DataType
except Exception as e:
    raise RuntimeError("pymilvus is required: pip install pymilvus") from e


class VectorStore:
    """
    Milvus-backed vector store exposing a FAISS-like interface:
      - add_with_ids(x, ids)
      - remove_ids(ids)
      - search(xq, k)
      - range_search(xq, radius, limit=None)

    Supports Milvus index types (e.g., FLAT, IVF_FLAT, IVF_PQ, IVF_SQ8, HNSW, SCANN, AUTOINDEX,
    BIN_* for binary, SPARSE_* for sparse). This class focuses on a single dense vector field.

    Note: Return shapes mimic FAISS: search -> (D, I) with shapes (nq, k).
    """

    def __init__(
        self,
        uri: str = "http://localhost:19530",
        collection_name: str = "vectors",
        dim: int = 384,
        metric_type: str = "L2",
        index_type: str = "FLAT",
        index_params: dict = None
    ):
        self.collection_name = collection_name
        self.index_type = index_type
        self.metric_type = metric_type
        self.id_field_name = "id"
        self.vector_field_name = "vector"
        self.index_name = "index"
        self.client = MilvusClient(uri=uri)
        
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
        
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True
        )
        schema.add_field(field_name=self.id_field_name, datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name=self.vector_field_name, datatype=DataType.FLOAT_VECTOR, dim=dim)
        self.client.create_collection(
            collection_name=self.collection_name, 
            schema=schema,
        )
        
        if index_params is None:
            index_params = self._default_index_params(index_type)
        
        _index_params = MilvusClient.prepare_index_params()
        _index_params.add_index(
            field_name=self.vector_field_name,
            metric_type=metric_type,
            index_type=index_type,
            index_name=self.index_name,
            params=index_params
        )
        self.client.create_index(
            collection_name=collection_name,
            index_params=_index_params,
            sync=True
        )
        self.load()

    # ----------------------- Public API (FAISS-like) -----------------------
    def add_with_ids(self, x: np.ndarray, ids: Union[np.ndarray, List[int]]) -> None:
        x = self._as_float32_2d(x)
        ids = self._as_int64_1d(ids)
        if len(ids) != len(x):
            raise ValueError(f"ids length {len(ids)} != vectors length {len(x)}")
        data = [{self.id_field_name: vid, self.vector_field_name: v} for (vid, v) in zip(ids, x)]
        self.client.insert(collection_name=self.collection_name, data=data)

    def remove_ids(self, ids: Union[np.ndarray, List[int]]) -> int:
        ids = self._as_int64_1d(ids)
        res = self.client.delete(collection_name=self.collection_name, ids=ids.tolist())
        count_removed = int(res.get("delete_count"))
        return count_removed

    def search(
        self,
        xq: np.ndarray,
        k: int = 10,
        search_params: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        xq = self._as_float32_2d(xq)
        if search_params is None:
            search_params = self._default_search_params(self.index_type, self.metric_type)
        res = self.client.search(
            collection_name=self.collection_name,
            data=xq.tolist(),
            limit=int(k),
            consistency_level="Strong",
        )
        I, D = self._hits_to_numpy(res, k=k)
        return D, I

    def range_search(
        self,
        xq: np.ndarray,
        radius: float,
        limit: Optional[int] = None,
        search_params: Optional[Dict] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Returns per-query variable-length results like FAISS range_search:
          - a list of distance arrays [dists_i]
          - a list of id arrays [ids_i]
        Milvus supports range search via `radius` and optional `range_filter`.
        We set a high limit to cap worst-case results when `limit` is None.
        """
        xq = self._as_float32_2d(xq)
        if search_params is None:
            search_params = self._default_search_params(self.index_type, self.metric_type)
        # Milvus requires `limit`; use a large default cap to avoid runaway results
        cap = int(limit) if limit is not None else 10000
        params = {"radius": float(radius)}
        params.update(search_params.get("params", {}))
        res = self.client.search(
            collection_name=self.collection_name,
            data=xq,
            anns_field=self.vector_field_name,
            limit=cap,
        )
        dists_list: List[np.ndarray] = []
        ids_list: List[np.ndarray] = []
        for hits in res:
            # Filter by radius client-side to strictly enforce radius (Milvus already does, but double-guard)
            ids = []
            dists = []
            for h in hits:
                if h.distance <= radius:
                    ids.append(h.id)
                    dists.append(h.distance)
            dists_list.append(np.asarray(dists, dtype=np.float32))
            ids_list.append(np.asarray(ids, dtype=np.int64))
        return dists_list, ids_list

    # --------------------------- Collection ops ----------------------------
    def load(self) -> None:
        self.client.load_collection(self.collection_name)
        self._loaded = True

    def release(self) -> None:
        self.client.release_collection(self.collection_name)
        self._loaded = False

    def drop(self) -> None:
        self.client.drop_collection(self.collection_name)
        self._loaded = False
    
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
        arr = np.asarray(list(ids) if not isinstance(ids, np.ndarray) else ids, dtype=np.int64)
        if arr.ndim != 1:
            raise ValueError("ids must be 1D")
        return arr

    @staticmethod
    def _hits_to_numpy(res, k: int) -> Tuple[np.ndarray, np.ndarray]:
        # Convert Milvus search results to (I, D) arrays (nq, k)
        nq = len(res)
        I = -np.ones((nq, k), dtype=np.int64)
        D = np.full((nq, k), np.inf, dtype=np.float32)
        for i, hits in enumerate(res):
            for j, h in enumerate(hits[:k]):
                I[i, j] = h.id
                D[i, j] = h.distance
        return I, D

    @staticmethod
    def _default_index_params(index_type: str) -> Dict:
        t = index_type.upper()
        if t == "FLAT":
            return {}
        if t in {"IVF_FLAT", "IVF_SQ8", "IVF_PQ"}:
            # Conservative defaults; tune per dataset
            base = {"nlist": 1024}
            if t == "IVF_PQ":
                base.update({"m": 16, "nbits": 8})  # PQ codebooks/precision
            return base
        if t == "HNSW":
            return {"M": 16, "efConstruction": 200}
        if t == "SCANN":
            # ScaNN params vary by build; provide a sane placeholder
            return {"nlist": 1024}
        if t == "AUTOINDEX":
            return {}
        # Fallback for GPU or newer types (e.g., GPU_IVF_FLAT, GPU_CAGRA)
        return {}

    @staticmethod
    def _default_search_params(index_type: str, metric_type: str) -> Dict:
        t = index_type.upper()
        # Milvus expects {"metric_type": ..., "params": {...}} at callsite; we return only "params" here
        if t in {"FLAT", "AUTOINDEX"}:
            return {"params": {}}
        if t in {"IVF_FLAT", "IVF_SQ8", "IVF_PQ", "SCANN"}:
            return {"params": {"nprobe": 16}}
        if t == "HNSW":
            return {"params": {"ef": 64}}
        return {"params": {}}


# ------------------------------- Example -----------------------------------
if __name__ == "__main__":
    dim = 64
    store = VectorStore(
        uri="http://localhost:19530",
        collection_name="demo_vectors",
        dim=dim,
        metric_type="L2",
        index_type="FLAT",
        index_params={"M": 16, "efConstruction": 100},
    )

    # Add data
    xb = np.random.random((1000, dim)).astype(np.float32)
    ids = np.arange(1000, dtype=np.int64)
    store.add_with_ids(xb, ids)

    # Search
    xq = np.random.random((5, dim)).astype(np.float32)
    D, I = store.search(xq, k=5)
    print("search distances shape:", D.shape, "ids shape:", I.shape)

    # Range search
    dists_list, ids_list = store.range_search(xq, radius=2.0, limit=50)
    print("range results lens:", [len(v) for v in ids_list])

    # Remove
    removed = store.remove_ids([0, 1, 2])
    print("removed:", removed)

    # Cleanup
    store.drop()
