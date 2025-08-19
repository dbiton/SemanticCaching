from collections import defaultdict
from typing import Optional, Dict, Set, Tuple
import faiss
import numpy as np

# Check about removing from index bug!
class OnlineClusters:
    def __init__(self, cluster_distance: float, dim: int):
        self.cluster_distance_sqrd = cluster_distance ** 2
        self.index = faiss.IndexIDMap2(faiss.IndexFlatL2(dim))
        self.vectors_ids_to_clusters_ids: Dict[int, int] = {}
        self.clusters_ids_to_vectors_ids: Dict[int, Set[int]] = defaultdict(set)

    def __get_cluster(self, vector: np.ndarray) -> Optional[int]:
        dists, ids = self.index.search(np.array([vector]), 1)
        if ids[0][0] == -1:
            return None
        if dists[0][0] < self.cluster_distance_sqrd:
            return ids[0][0]
        else:
            return None

    def remove_vector(self, vector_id: int) -> None:
        cluster_id = self.vectors_ids_to_clusters_ids.pop(vector_id)
        self.clusters_ids_to_vectors_ids[cluster_id].pop(vector_id)
        ids = np.array([vector_id], dtype=np.int64)
        self.index.remove_ids(ids)

    def add_vector(self, vector: np.ndarray, vector_id: int) -> int:
        cluster_id = self.__get_cluster(vector)
        if cluster_id is None:
            self.index.add_with_ids(np.array([vector]), np.array([vector_id]))
            cluster_id = vector_id
        self.vectors_ids_to_clusters_ids[vector_id] = cluster_id
        self.clusters_ids_to_vectors_ids[cluster_id].add(vector_id)
        return cluster_id
    
    def get_cluster(self, vector_id: int) -> int:
        return self.vectors_ids_to_clusters_ids[vector_id]

    def pop_vector(self, cluster_id: int) -> Tuple[int, bool]:
        vector_id = self.clusters_ids_to_vectors_ids[cluster_id].pop()
        cluster_emptied = False
        if len(self.clusters_ids_to_vectors_ids[cluster_id]) == 0:
            self.clusters_ids_to_vectors_ids.pop(cluster_id)
            cluster_emptied = True
        cluster_id = self.vectors_ids_to_clusters_ids.pop(vector_id)
        ids = np.array([vector_id], dtype=np.int64)
        self.index.remove_ids(ids)
        return vector_id, cluster_emptied