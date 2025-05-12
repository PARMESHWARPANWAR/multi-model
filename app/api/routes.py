import logging
import os
import uuid
import json
import tempfile
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends, Request, UploadFile, File, Form, HTTPException
from app.embedding.service import EmbeddingService

logger = logging.getLogger(__name__)

router = APIRouter()

embedding_service = EmbeddingService()

# Dependency to get storage service from app state
async def get_storage_service(request: Request):
    if not hasattr(request.app.state, "storage_service"):
        logger.warning("Storage service not found in app state, using global instance")
        from app.storage.service import StorageService
        return StorageService(use_mock_if_unavailable=True)
    return request.app.state.storage_service

# Dependency to get embedding service (for future use)
async def get_embedding_service(request: Request):
    if hasattr(request.app.state, "embedding_service"):
        return request.app.state.embedding_service
    return embedding_service

@router.get("/categories")
async def get_categories(storage_service = Depends(get_storage_service)):
    """Get list of available categories"""
    try:
        if storage_service is None or storage_service.db is None:
            return {"categories": ["general", "nature", "technology", "art"]}

        categories = storage_service.db.metadata.distinct("category")
        return {"categories": categories if categories else []}
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        return {"categories": ["general", "nature", "technology", "art"], "error": str(e)}

@router.get("/modalities")
async def get_modalities(storage_service = Depends(get_storage_service)):
    """Get list of available modalities"""
    try:
        if storage_service is None or storage_service.db is None:
            return {"modalities": ["text", "image", "audio"]}

        modalities = storage_service.db.metadata.distinct("modality")
        print("modalities in db", modalities)
        return {"modalities": modalities if modalities else ["text", "image", "audio"]}
    except Exception as e:
        logger.error(f"Error getting modalities: {e}")
        return {"modalities": ["text", "image", "audio"], "error": str(e)}

@router.get("/stats")
async def get_stats(storage_service = Depends(get_storage_service)):
    try:
        if storage_service is None or storage_service.db is None:
            return {
                "total_items": 0,
                "modality_counts": {"text": 0, "image": 0, "audio": 0, "video": 0},
                "category_counts": {},
                "status": "database_unavailable"
            }
        
        total_items = storage_service.db.metadata.count_documents({})
        modality_counts = {}
        
        for modality in ["text", "image", "audio", "video"]:
            count = storage_service.db.metadata.count_documents({"modality": modality})
            modality_counts[modality] = count

        category_counts = {}
        categories = storage_service.db.metadata.distinct("category")
        
        for category in categories:
            count = storage_service.db.metadata.count_documents({"category": category})
            category_counts[category] = count
        
        return {
            "total_items": total_items,
            "modality_counts": modality_counts,
            "category_counts": category_counts,
            "status": "ok"
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {
            "total_items": 0,
            "modality_counts": {"text": 0, "image": 0, "audio": 0, "video": 0},
            "category_counts": {},
            "status": "error",
            "error": str(e)
        }

@router.get("/search/text")
async def search_by_text_get(
    query: str,
    k: int = 10,
    category: Optional[str] = None,
    modality: Optional[str] = None,
    embedding_service = Depends(get_embedding_service),
    storage_service = Depends(get_storage_service)
):
    try:

        result = await embedding_service.process_text([query])
        query_embedding = result["embeddings"]

        filters = {}
        if category:
            filters["category"] = category
        if modality:
            filters["modality"] = modality
 
        if filters:
            search_results = storage_service.filter_search(query_embedding, filters, k)
        else:
            search_results = storage_service.search(query_embedding, k)
  
        return {
            "query": query,
            "filters": filters,
            "results": search_results
        }
    except Exception as e:
        logger.error(f"Error in text search: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@router.post("/search/image")
async def search_by_image(
    file: UploadFile = File(...),
    k: int = 10,
    category: Optional[str] = None,
    modality: Optional[str] = None,
    embedding_service = Depends(get_embedding_service),
    storage_service = Depends(get_storage_service)
):
    
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}.jpg")
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        result = await embedding_service.process_images([temp_path])
        query_embedding = result["embeddings"]

        filters = {}
        if category:
            filters["category"] = category
        if modality:
            filters["modality"] = modality
        
        if filters:
            search_results = storage_service.filter_search(query_embedding, filters, k)
        else:
            search_results = storage_service.search(query_embedding, k)
        
        return {
            "filters": filters,
            "results": search_results
        }
    except Exception as e:
        logger.error(f"Error in image search: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/search/audio")
async def search_by_audio(
    file: UploadFile = File(...),
    k: int = 10,
    category: Optional[str] = None,
    modality: Optional[str] = None,
    embedding_service = Depends(get_embedding_service),
    storage_service = Depends(get_storage_service)
):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    print('called audio search 12')
    
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        result = await embedding_service.process_audio([temp_path])
        query_embedding = result["embeddings"]
        
        filters = {}
        filters = {"modality": "audio"}
        
        if category:
            filters["category"] = category
        # if modality:
        #     filters["modality"] = modality
        
        # Search with filters
        if filters:
            search_results = storage_service.filter_search(query_embedding, filters, k)
        else:
            search_results = storage_service.search(query_embedding, k)
        
        # Return search results
        return {
            "filters": filters,
            "results": search_results
        }
    except Exception as e:
        logger.error(f"Error in audio search: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    
    
       