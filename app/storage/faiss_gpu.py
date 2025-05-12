import faiss
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

class FaissGPUIndex:
    """
    FAISS index with GPU acceleration for faster similarity search
    """
    def __init__(self, dimension: int = 1024, use_gpu: bool = True, gpu_id: int = 0):
        self.dimension = dimension
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        self.gpu_id = gpu_id
        
        # Create base index (initially on CPU)
        self.cpu_index = faiss.IndexFlatIP(dimension)  # Inner product similarity (cosine)
        
        # Transfer to GPU if available
        if self.use_gpu:
            self.gpu_resources = faiss.StandardGpuResources()
            gpu_options = faiss.GpuIndexFlatConfig()
            gpu_options.device = gpu_id
            
            # Transfer index to GPU
            self.index = faiss.index_cpu_to_gpu(self.gpu_resources, gpu_id, self.cpu_index)
            print(f"FAISS index running on GPU {gpu_id}")
        else:
            self.index = self.cpu_index
            print("FAISS index running on CPU (GPU not available)")
    
    def add(self, vectors: np.ndarray):
        """
        Add vectors to index
        
        Args:
            vectors: Vectors to add, shape (n, dimension)
        """
        # Ensure vectors are in the correct format
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)
        
        assert vectors.shape[1] == self.dimension, f"Expected dimension {self.dimension}, got {vectors.shape[1]}"
        
        # Normalize for cosine similarity
        faiss.normalize_L2(vectors)
        
        # Add to index
        self.index.add(vectors)
        
        # If using GPU, sync changes back to CPU index for persistence
        if self.use_gpu:
            # Convert GPU index back to CPU for saving
            faiss.index_gpu_to_cpu(self.index, self.cpu_index)
    
    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors
        
        Args:
            query: Query vector, shape (1, dimension)
            k: Number of results to return
        
        Returns:
            Tuple of (distances, indices)
        """
        # Ensure query is in the correct format
        if len(query.shape) == 1:
            query = query.reshape(1, -1)
        
        assert query.shape[1] == self.dimension, f"Expected dimension {self.dimension}, got {query.shape[1]}"
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query)
        
        # Search
        distances, indices = self.index.search(query, k)
        
        return distances, indices
    
    def batch_search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors in batch
        
        Args:
            queries: Query vectors, shape (n, dimension)
            k: Number of results to return per query
        
        Returns:
            Tuple of (distances, indices), shapes (n, k)
        """
        # Ensure queries are in the correct format
        if len(queries.shape) == 1:
            queries = queries.reshape(1, -1)
        
        assert queries.shape[1] == self.dimension, f"Expected dimension {self.dimension}, got {queries.shape[1]}"
        
        # Normalize for cosine similarity
        faiss.normalize_