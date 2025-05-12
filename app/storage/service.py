import os
import logging
import faiss
import numpy as np
import json
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from app.storage.models import MetadataModel, EmbeddingDocument

DB_URL = 'mongodb://localhost:27017/'

logger = logging.getLogger(__name__)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class StorageService:
    def __init__(self, mongo_uri=DB_URL, use_mock_if_unavailable=False):
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
            
        self.index_dimension = 1024
        try:
            self.index = faiss.IndexFlatIP(self.index_dimension)
            logger.info("Successfully initialized FAISS index")
      
            os.makedirs("data/indices", exist_ok=True)
          
            if os.path.exists("data/indices/faiss_index.bin"):
                try:
                    self.load_index()
                    logger.info("Successfully loaded FAISS index from disk")
                except Exception as e:
                    logger.error(f"Error loading FAISS index: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            self.index = None
    
    def store_metadata(self, metadata: MetadataModel) -> str:
        if self.metadata_collection is None:
            raise ValueError("MongoDB not available")
            
        result = self.metadata_collection.insert_one(metadata.dict())
        return str(result.inserted_id)
          
    def store_embedding(self, item_id: str, metadata_id: str, embedding: np.ndarray, modality: str) -> str:
        if self.index is None or self.embedding_collection is None:
            raise ValueError("FAISS index or MongoDB not available")
            
        if len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)

        current_index = self.index.ntotal

        embedding_copy = embedding.copy()
        faiss.normalize_L2(embedding_copy)
        self.index.add(embedding_copy)
        
        embedding_doc = EmbeddingDocument(
            id=str(current_index),
            metadata_id=metadata_id,
            modality=modality,
            dimension=embedding.shape[1]
        )
        
        doc_dict = embedding_doc.dict()
        doc_dict['original_id'] = item_id
        
        result = self.embedding_collection.insert_one(doc_dict)

        self.save_index()
        
        return str(result.inserted_id)
    
    def batch_store_embeddings(self, item_ids: List[str], metadata_ids: List[str], 
                               embeddings: np.ndarray, modality: str) -> List[str]:
       
        if self.index is None or self.embedding_collection is None:
            raise ValueError("FAISS index or MongoDB not available")
 
        start_index = self.index.ntotal   

        embeddings_norm = embeddings.copy()
        faiss.normalize_L2(embeddings_norm)
        self.index.add(embeddings_norm)

        embedding_docs = []
        for i, (item_id, metadata_id) in enumerate(zip(item_ids, metadata_ids)):
            faiss_index = start_index + i
            embedding_doc = EmbeddingDocument(
                id=str(faiss_index),
                metadata_id=metadata_id,
                modality=modality,
                dimension=embeddings.shape[1]
            )

            doc_dict = embedding_doc.dict()
            doc_dict['original_id'] = item_id 
            embedding_docs.append(doc_dict)
        
        result = self.embedding_collection.insert_many(embedding_docs)

        self.save_index()
        
        return [str(id) for id in result.inserted_ids]
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        if self.index is None:
            logger.error("FAISS index not available")
            return []
   
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding_norm = query_embedding.copy()
        faiss.normalize_L2(query_embedding_norm)
        
        k_actual = min(k, self.index.ntotal)
        if k_actual == 0:
            return []
    
        distances, indices = self.index.search(query_embedding_norm, k_actual)
        
        print(f"FAISS returned indices: {indices[0]}")
        print(f"FAISS returned distances: {distances[0]}")
    
        results = []
        
        if self.embedding_collection is None or self.metadata_collection is None:
            logger.error("MongoDB not available for search metadata retrieval")
            return []
        
        for i, idx in enumerate(indices[0]):
            if idx < 0:
                continue

            try:
                embedding_doc = self.embedding_collection.find_one({"id": str(idx)})
                if embedding_doc:
                    metadata_id = embedding_doc["metadata_id"]
                    
                    from bson import ObjectId
                    if isinstance(metadata_id, str):
                        metadata_id = ObjectId(metadata_id)
                        
                    metadata = self.metadata_collection.find_one({"_id": metadata_id})
                    
                    if metadata:
                        metadata['_id'] = str(metadata['_id'])
                        results.append({
                            "metadata": metadata,
                            "score": float(distances[0][i])
                        })
            except Exception as e:
                logger.error(f"Error retrieving metadata for search result: {e}")
                print(f"Error details: {e}")
        return results
    
    def filter_search(self, query_embedding: np.ndarray, filters: Dict[str, Any], k: int = 10) -> List[Dict[str, Any]]:
        if self.index is None:
            logger.error("FAISS index not available")
            return []
        
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding_norm = query_embedding.copy()
        faiss.normalize_L2(query_embedding_norm)
        
        try:
            k_search = min(k * 4, self.index.ntotal)
            distances, indices = self.index.search(query_embedding_norm, k_search)
        except Exception as e:
            logger.error(f"FAISS search error: {e}")
            return []

        results = []
        
        if self.embedding_collection is None or self.metadata_collection is None:
            logger.error("MongoDB not available for filtered search")
            return []
        
        for i, idx in enumerate(indices[0]):
            if idx < 0: 
                continue

            try:
                embedding_doc = self.embedding_collection.find_one({"id": str(idx)})
                if embedding_doc:
                    metadata_id = embedding_doc["metadata_id"]
                    from bson import ObjectId
                    if isinstance(metadata_id, str):
                        metadata_id = ObjectId(metadata_id)
                    
                    metadata = self.metadata_collection.find_one({"_id": metadata_id})

                    if metadata:
                        include = True
                        for key, value in filters.items():
                            if key in metadata and metadata[key] != value:
                                include = False
                                break
                        if include:
                            metadata['_id'] = str(metadata['_id'])
                            results.append({
                                "metadata": metadata,
                                "score": float(distances[0][i])
                            })
            except Exception as e:
                logger.error(f"Error applying filters in search: {e}")

        return results[:k]


    def save_index(self):
        """Save FAISS index to disk"""
        if self.index is None:
            logger.error("Cannot save FAISS index: index not initialized")
            return
            
        try:
            faiss.write_index(self.index, "data/indices/faiss_index.bin")
            logger.info("FAISS index saved successfully")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
    
    def load_index(self):
        """Load FAISS index from disk"""
        try:
            if os.path.exists("data/indices/faiss_index.bin"):
                self.index = faiss.read_index("data/indices/faiss_index.bin")
                logger.info(f"FAISS index loaded with {self.index.ntotal} vectors")
            else:
                logger.warning("No FAISS index file found to load")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            raise
                