from __future__ import annotations
import numpy as np
import zlib 

# Try to import mmh3 for speed, fallback to zlib if missing
try:
    import mmh3
except ImportError:
    mmh3 = None

class DensityEstimator:
    """
    Robust RACE with Bias Correction and Median Aggregation.
    Fixes the "High MAE" issue by subtracting background noise.
    """

    def __init__(
        self,
        dim: int,
        reps: int = 50,           # R: Lower is often fine if using Median
        hashes_per_rep: int = 4,  # k: LSH concatenation
        width: int = 20003,       # W: USE A PRIME NUMBER. Reduces harmonic collisions.
        cell_size: float = 1.0,   
        seed: int = 42
    ):
        self.dim = dim
        self.reps = reps
        self.k = hashes_per_rep
        self.width = width
        self.cell_size = cell_size
        
        # The Sketch: R x W
        self.sketch = np.zeros((reps, width), dtype=np.int32)
        
        # LSH params
        rng = np.random.default_rng(seed)
        self.A = rng.standard_normal((reps, self.k, dim)).astype(np.float32)
        self.B = rng.uniform(0.0, cell_size, size=(reps, self.k)).astype(np.float32)
        
        # Seeds for universal hashing
        self.hash_seeds = rng.integers(0, 2**32, size=reps)
        
        # Track total items for Bias Correction
        self.n_seen = 0

    def _hash_to_indices(self, x: np.ndarray) -> np.ndarray:
        # 1. LSH Projection (R, k)
        proj = (self.A @ x) + self.B
        quantized = np.floor(proj / self.cell_size).astype(np.int32)
        
        # 2. Map to sketch columns [0, W)
        indices = np.zeros(self.reps, dtype=np.int32)
        
        # Optimization: Pre-bind standard library functions
        lb_hash = zlib.crc32
        
        for r in range(self.reps):
            row_data = quantized[r].tobytes()
            
            if mmh3:
                # Fast path
                h = mmh3.hash(row_data, seed=int(self.hash_seeds[r]))
            else:
                # Fallback (Slower but works)
                # Combine data + seed manually for randomness
                h = lb_hash(row_data) + self.hash_seeds[r]
                
            indices[r] = h % self.width
            
        return indices

    def auto_tune(self, sample: np.ndarray):
        """
        Calculates optimal cell_size based on median pairwise distance.
        """
        from scipy.spatial.distance import pdist
        # Subsample if large
        if len(sample) > 1000:
            indices = np.random.choice(len(sample), 1000, replace=False)
            sample = sample[indices]
            
        dists = pdist(sample, 'euclidean')
        # Using 30th percentile is a safe bet for "local" density
        # If correlation is 0, try increasing this to 40 or 50.
        new_w = float(np.percentile(dists, 30))
        
        if new_w < 1e-6: new_w = 1.0 # Safety
        
        print(f"Auto-tune: cell_size set to {new_w:.4f}")
        self.cell_size = new_w
        
        # Reset projections with new size
        rng = np.random.default_rng(42)
        self.B = rng.uniform(0.0, self.cell_size, size=(self.reps, self.k)).astype(np.float32)
        self.sketch.fill(0)
        self.n_seen = 0

    def update(self, batch: np.ndarray):
        X = np.asarray(batch, dtype=np.float32)
        for x in X:
            indices = self._hash_to_indices(x)
            # Update sketch
            self.sketch[np.arange(self.reps), indices] += 1
            self.n_seen += 1

    def query(self, batch: np.ndarray) -> np.ndarray:
        X = np.asarray(batch, dtype=np.float32)
        densities = np.zeros(X.shape[0], dtype=np.float32)
        
        # BIAS CORRECTION TERM
        # This is the expected "background noise" in any bucket
        # If W is large, noise is small. If W is small, noise is high.
        noise_floor = self.n_seen / self.width
        
        for i, x in enumerate(X):
            indices = self._hash_to_indices(x)
            counts = self.sketch[np.arange(self.reps), indices]
            
            # Use MEDIAN to ignore outlier collisions
            raw_est = np.median(counts)
            
            # Subtract noise
            corrected_est = max(0.0, raw_est - noise_floor)
            
            densities[i] = corrected_est
            
        return densities