import faiss
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import os
import json

class FaissGPUIndex:
    def __init__(self, dimension: int = 1024, use_gpu: bool = True, gpu_id: int = 0):
        self.dimension = dimension
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        self.gpu_id = gpu_id
    
        self.cpu_index = faiss.IndexFlatIP(dimension)
      
        if self.use_gpu:
            self.gpu_resources = faiss.StandardGpuResources()
            gpu_options = faiss.GpuIndexFlatConfig()
            gpu_options.device = gpu_id
            
            self.index = faiss.index_cpu_to_gpu(self.gpu_resources, gpu_id, self.cpu_index)
            print(f"FAISS index running on GPU {gpu_id}")
        else:
            self.index = self.cpu_index
            print("FAISS index running on CPU (GPU not available)")
    
    def add(self, vectors: np.ndarray):
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)
        
        assert vectors.shape[1] == self.dimension, f"Expected dimension {self.dimension}, got {vectors.shape[1]}"
       
        vectors = vectors.copy()
        faiss.normalize_L2(vectors)
       
        self.index.add(vectors)
       
        if self.use_gpu:
            self.cpu_index.add(vectors)
    
    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if len(query.shape) == 1:
            query = query.reshape(1, -1)
        
        assert query.shape[1] == self.dimension, f"Expected dimension {self.dimension}, got {query.shape[1]}"
     
        query = query.copy()
        faiss.normalize_L2(query)
      
        distances, indices = self.index.search(query, k)
        
        return distances, indices
    
    def save(self, file_path: str):
        if self.use_gpu:
            faiss.write_index(self.cpu_index, file_path)
        else:
            faiss.write_index(self.index, file_path)
    
    def load(self, file_path: str):
        self.cpu_index = faiss.read_index(file_path)

        if self.use_gpu:
            self.index = faiss.index_cpu_to_gpu(self.gpu_resources, self.gpu_id, self.cpu_index)
        else:
            self.index = self.cpu_index
    
    def batch_search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if len(queries.shape) == 1:
            queries = queries.reshape(1, -1)
        
        assert queries.shape[1] == self.dimension, f"Expected dimension {self.dimension}, got {queries.shape[1]}"
        
        # Normalize for cosine similarity
        queries = queries.copy() 
        faiss.normalize_L2(queries)
        
        # Search
        distances, indices = self.index.search(queries, k)
        
        return distances, indices
    
    def get_index_stats(self) -> Dict[str, Any]:
        return {
            'dimension': self.dimension,
            'total_vectors': self.index.ntotal,
            'on_gpu': self.use_gpu,
            'gpu_id': self.gpu_id if self.use_gpu else None
        }

class OptimizedVectorStorage:
    def __init__(self, dimension: int = 1024, use_gpu: bool = True):
        self.dimension = dimension
        self.faiss_index = FaissGPUIndex(dimension, use_gpu)
        self.id_mapping = {}  
        self.reverse_mapping = {} 
        self.next_index = 0
        
    def add_item(self, id: str, embedding: np.ndarray, metadata: Dict[str, Any] = None):
        self.faiss_index.add(embedding)
       
        self.id_mapping[self.next_index] = {
            'id': id,
            'metadata': metadata or {}
        }
        self.reverse_mapping[id] = self.next_index
        
        self.next_index += 1
        
    def add_batch(self, ids: List[str], embeddings: np.ndarray, metadatas: List[Dict[str, Any]] = None):
        self.faiss_index.add(embeddings)
        
        for i, id in enumerate(ids):
            metadata = metadatas[i] if metadatas else {}
            faiss_idx = self.next_index + i
            
            self.id_mapping[faiss_idx] = {
                'id': id,
                'metadata': metadata
            }
            self.reverse_mapping[id] = faiss_idx
       
        self.next_index += len(ids)
        
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        distances, indices = self.faiss_index.search(query_embedding, k)
       
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            # Skip invalid indices (FAISS returns -1 for padded results)
            if idx < 0:
                continue
                
            # Get item data
            item_data = self.id_mapping.get(int(idx))
            if item_data:
                results.append({
                    'id': item_data['id'],
                    'score': float(distance),  # Convert to Python float
                    'metadata': item_data['metadata']
                })
        
        return results
        
    def batch_search(self, query_embeddings: np.ndarray, k: int = 10) -> List[List[Dict[str, Any]]]:
        distances, indices = self.faiss_index.batch_search(query_embeddings, k)
        
        # Convert to result format
        all_results = []
        for query_idx in range(len(query_embeddings)):
            results = []
            for result_idx in range(k):
                idx = indices[query_idx][result_idx]
                # Skip invalid indices
                if idx < 0:
                    continue
                    
                # Get item data
                item_data = self.id_mapping.get(int(idx))
                if item_data:
                    results.append({
                        'id': item_data['id'],
                        'score': float(distances[query_idx][result_idx]),
                        'metadata': item_data['metadata']
                    })
            all_results.append(results)
        
        return all_results
    
    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        self.faiss_index.save(os.path.join(directory, 'index.faiss'))
        
        # Save mappings
        with open(os.path.join(directory, 'id_mapping.json'), 'w') as f:
            # Convert keys to strings for JSON serialization
            serializable_mapping = {str(k): v for k, v in self.id_mapping.items()}
            json.dump(serializable_mapping, f)
            
        with open(os.path.join(directory, 'reverse_mapping.json'), 'w') as f:
            # Convert values to strings for JSON serialization
            serializable_mapping = {k: str(v) for k, v in self.reverse_mapping.items()}
            json.dump(serializable_mapping, f)
            
        # Save metadata
        with open(os.path.join(directory, 'metadata.json'), 'w') as f:
            json.dump({
                'dimension': self.dimension,
                'next_index': self.next_index
            }, f)
    
    def load(self, directory: str):
        self.faiss_index.load(os.path.join(directory, 'index.faiss'))
        
        # Load mappings
        with open(os.path.join(directory, 'id_mapping.json'), 'r') as f:
            serialized_mapping = json.load(f)
            # Convert keys back to integers
            self.id_mapping = {int(k): v for k, v in serialized_mapping.items()}
            
        with open(os.path.join(directory, 'reverse_mapping.json'), 'r') as f:
            serialized_mapping = json.load(f)
            # Convert values back to integers
            self.reverse_mapping = {k: int(v) for k, v in serialized_mapping.items()}
            
        # Load metadata
        with open(os.path.join(directory, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            self.dimension = metadata['dimension']
            self.next_index = metadata['next_index']