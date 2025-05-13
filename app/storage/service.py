import os
import time
import json
import faiss
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from pymongo import MongoClient, ASCENDING, TEXT
from bson import ObjectId
from app.storage.models import MetadataModel, EmbeddingDocument
from app.storage.faiss_gpu import FaissGPUIndex, OptimizedVectorStorage

logger = logging.getLogger(__name__)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

DB_URL = os.getenv('MONGODB_URL', 'mongodb://localhost:27017/multi_modal_search')

class StorageService:
    def __init__(self, mongo_uri=DB_URL, use_mock_if_unavailable=False):
        # Initialize MongoDB connection with fallback
        try:
            self.client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            self.db = self.client.multimodal_search
            self.metadata_collection = self.db.metadata
            self.embedding_collection = self.db.embeddings
            
            # Test connection
            self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            
            if use_mock_if_unavailable:
                try:
                    logger.info("Using in-memory mock MongoDB")
                    import mongomock
                    self.client = mongomock.MongoClient()
                    self.db = self.client.multimodal_search
                    self.metadata_collection = self.db.metadata
                    self.embedding_collection = self.db.embeddings
                except ImportError:
                    logger.error("mongomock not installed, cannot use mock MongoDB")
                    self.client = None
                    self.db = None
                    self.metadata_collection = None
                    self.embedding_collection = None
            else:
                self.client = None
                self.db = None
                self.metadata_collection = None
                self.embedding_collection = None
        
        # Initialize GPU-accelerated vector storage
        self.index_dimension = 1024
        try:
            use_gpu = faiss.get_num_gpus() > 0
            self.vector_storage = OptimizedVectorStorage(dimension=self.index_dimension, use_gpu=use_gpu)
            logger.info(f"Successfully initialized vector storage (GPU: {use_gpu})")
            
            # Create index directory
            os.makedirs("data/indices", exist_ok=True)
            
            # Load existing index if available
            if os.path.exists("data/indices/index.faiss"):
                try:
                    self.vector_storage.load("data/indices")
                    logger.info(f"Successfully loaded vector storage with {self.vector_storage.next_index} items")
                except Exception as e:
                    logger.error(f"Error loading vector storage: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize vector storage: {e}")
            self.vector_storage = None
        
        # Create indexes for faster queries if MongoDB is available
        if self.metadata_collection is not None and self.embedding_collection is not None:    
            self._create_indexes()
        
        # Create cache for frequently accessed metadata
        self.metadata_cache = {}
        self.cache_size = 10000  # Maximum number of items in cache
    
    def _create_indexes(self):
        """Create MongoDB indexes for faster queries"""
        try:
            # Create text index for full text search
            self.metadata_collection.create_index([("title", TEXT), ("description", TEXT)])
            
            # Create index for category and modality for filtering
            self.metadata_collection.create_index([("category", ASCENDING), ("modality", ASCENDING)])
            
            # Create compound index for common queries
            self.metadata_collection.create_index([("modality", ASCENDING), ("created_at", -1)])
            
            # Create index for embedding lookups
            self.embedding_collection.create_index("metadata_id")
            self.embedding_collection.create_index("id", unique=True)
            
            logger.info("Created MongoDB indexes successfully")
        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")
    
    def store_metadata(self, metadata: MetadataModel) -> str:
        """Store metadata in MongoDB"""
        if self.metadata_collection is None:
            raise ValueError("MongoDB not available")
        
        # Convert to dictionary
        metadata_dict = metadata.dict()
        
        # Store in MongoDB
        result = self.metadata_collection.insert_one(metadata_dict)
        inserted_id = str(result.inserted_id)
        
        # Add to cache if small enough
        if len(metadata_dict['id']) < 64:  # Only cache small IDs
            self.metadata_cache[metadata.id] = metadata_dict
            
            # Trim cache if needed
            if len(self.metadata_cache) > self.cache_size:
                # Remove oldest (first) item
                first_key = next(iter(self.metadata_cache))
                del self.metadata_cache[first_key]
        
        return inserted_id
    
    def store_embedding(self, item_id: str, metadata_id: str, embedding: np.ndarray, modality: str) -> str:
        """Store embedding in vector storage and reference in MongoDB"""
        if self.vector_storage is None or self.embedding_collection is None:
            raise ValueError("Vector storage or MongoDB not available")
        
        # Ensure embedding is 1D
        if len(embedding.shape) == 2:
            embedding = embedding.squeeze()
        
        # Add to vector storage with metadata
        metadata = {"metadata_id": metadata_id, "modality": modality}
        self.vector_storage.add_item(item_id, embedding, metadata)
        
        # Create embedding document for MongoDB
        embedding_doc = EmbeddingDocument(
            id=item_id,
            metadata_id=metadata_id,
            modality=modality,
            dimension=embedding.shape[0]
        )
        
        # Store in MongoDB
        result = self.embedding_collection.insert_one(embedding_doc.dict())
        
        # Save vector storage periodically
        if self.vector_storage.next_index % 1000 == 0:
            self.save_index()
        
        return str(result.inserted_id)
    
    def batch_store_embeddings(self, item_ids: List[str], metadata_ids: List[str], 
                             embeddings: np.ndarray, modality: str) -> List[str]:
        """Store multiple embeddings in batch"""
        if self.vector_storage is None or self.embedding_collection is None:
            raise ValueError("Vector storage or MongoDB not available")
        
        start_time = time.time()
        batch_size = len(item_ids)
        
        # Prepare metadata for vector storage
        metadatas = [{"metadata_id": mid, "modality": modality} for mid in metadata_ids]
        
        # Add to vector storage
        self.vector_storage.add_batch(item_ids, embeddings, metadatas)
        
        # Create embedding documents for MongoDB
        embedding_docs = []
        for i, (item_id, metadata_id) in enumerate(zip(item_ids, metadata_ids)):
            embedding_doc = EmbeddingDocument(
                id=item_id,
                metadata_id=metadata_id,
                modality=modality,
                dimension=embeddings.shape[1]
            )
            embedding_docs.append(embedding_doc.dict())
        
        # Store in MongoDB
        result = self.embedding_collection.insert_many(embedding_docs)
        
        # Save vector storage
        self.save_index()
        
        # Log performance
        elapsed_time = time.time() - start_time
        logger.info(f"Batch stored {batch_size} embeddings in {elapsed_time:.2f}s")
        
        return [str(id) for id in result.inserted_ids]
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar items using GPU-accelerated FAISS"""
        if self.vector_storage is None:
            logger.error("Vector storage not available")
            return []
        
        start_time = time.time()
        
        # Ensure query is 1D
        if len(query_embedding.shape) == 2:
            query_embedding = query_embedding.squeeze()
        
        # Search using vector storage
        vector_results = self.vector_storage.search(query_embedding, k)
        
        # Enhance results with full metadata from MongoDB
        results = []
        
        if self.metadata_collection is None:
            logger.error("MongoDB not available for search metadata retrieval")
            return []
        
        for result in vector_results:
            item_id = result['id']
            metadata_id = result['metadata'].get('metadata_id')
            
            try:
                # Check cache first
                cache_key = f"meta_{metadata_id}"
                if cache_key in self.metadata_cache:
                    metadata = self.metadata_cache[cache_key]
                else:
                    # Get from MongoDB
                    if isinstance(metadata_id, str):
                        metadata_id = ObjectId(metadata_id)
                    
                    metadata = self.metadata_collection.find_one({"_id": metadata_id})
                    
                    # Cache result
                    if metadata:
                        metadata['_id'] = str(metadata['_id'])
                        self.metadata_cache[cache_key] = metadata
                
                if metadata:
                    results.append({
                        "metadata": metadata,
                        "score": result['score']
                    })
            except Exception as e:
                logger.error(f"Error retrieving metadata for search result: {e}")
        
        # Log performance
        elapsed_time = time.time() - start_time
        logger.info(f"Search completed in {elapsed_time:.2f}s, found {len(results)} results")
        
        return results
    
    def filter_search(self, query_embedding: np.ndarray, filters: Dict[str, Any], k: int = 10) -> List[Dict[str, Any]]:
        if self.vector_storage is None:
            logger.error("Vector storage not available")
            return []
        
        start_time = time.time()

        if len(query_embedding.shape) == 2:
            query_embedding = query_embedding.squeeze()
     
        k_search = min(k * 4, self.vector_storage.next_index)
        vector_results = self.vector_storage.search(query_embedding, k_search)
    
        results = []
        
        if self.metadata_collection is None:
            logger.error("MongoDB not available for filtered search")
            return []
        
        for result in vector_results:
            metadata_id = result['metadata'].get('metadata_id')
            
            try:
                if isinstance(metadata_id, str):
                    metadata_id = ObjectId(metadata_id)
                
                # Apply filters in MongoDB query
                query = {"_id": metadata_id}
                for key, value in filters.items():
                    if value is not None:
                        query[key] = value
                
                # Get filtered metadata
                metadata = self.metadata_collection.find_one(query)
                
                if metadata:
                    metadata['_id'] = str(metadata['_id'])
                    results.append({
                        "metadata": metadata,
                        "score": result['score']
                    })
                    
                    # Stop if we have enough results
                    if len(results) >= k:
                        break
            except Exception as e:
                logger.error(f"Error applying filters in search: {e}")
        
        # Log performance
        elapsed_time = time.time() - start_time
        logger.info(f"Filtered search completed in {elapsed_time:.2f}s, found {len(results)} results")
        
        return results[:k]
    
    def save_index(self):
        """Save vector storage to disk"""
        if self.vector_storage is None:
            logger.error("Cannot save vector storage: not initialized")
            return
        
        try:
            self.vector_storage.save("data/indices")
            logger.info(f"Vector storage saved successfully with {self.vector_storage.next_index} vectors")
        except Exception as e:
            logger.error(f"Error saving vector storage: {e}")
    
    def load_index(self):
        """Load vector storage from disk"""
        try:
            if os.path.exists("data/indices/index.faiss"):
                self.vector_storage.load("data/indices")
                logger.info(f"Vector storage loaded with {self.vector_storage.next_index} vectors")
            else:
                logger.warning("No vector storage files found to load")
        except Exception as e:
            logger.error(f"Error loading vector storage: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        stats = {
            "cache_size": len(self.metadata_cache),
            "cache_max_size": self.cache_size
        }
        
        if self.vector_storage:
            vector_stats = self.vector_storage.faiss_index.get_index_stats()
            stats.update({
                "index_vectors": vector_stats['total_vectors'],
                "index_dimension": vector_stats['dimension'],
                "gpu_enabled": vector_stats['on_gpu'],
                "gpu_id": vector_stats['gpu_id']
            })
        
        if self.metadata_collection is not None and self.embedding_collection is not None:
            stats.update({
                "metadata_count": self.metadata_collection.count_documents({}),
                "embedding_count": self.embedding_collection.count_documents({})
            })
        
        return stats