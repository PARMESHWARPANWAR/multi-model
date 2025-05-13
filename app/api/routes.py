import logging
import os
import uuid
import json
import tempfile
import time
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends, Request, UploadFile, File, Form, HTTPException
from app.embedding.service import EmbeddingService

logger = logging.getLogger(__name__)

router = APIRouter()

embedding_service = EmbeddingService()

async def get_storage_service(request: Request):
    if not hasattr(request.app.state, "storage_service"):
        logger.warning("Storage service not found in app state, using global instance")
        from app.storage.service import StorageService
        return StorageService(use_mock_if_unavailable=True)
    return request.app.state.storage_service

async def get_embedding_service(request: Request):
    if hasattr(request.app.state, "embedding_service"):
        return request.app.state.embedding_service
    return embedding_service

async def get_performance_monitor(request: Request):
    if hasattr(request.app.state, "performance_monitor"):
        return request.app.state.performance_monitor
    return None

async def get_metrics(request: Request):
    if hasattr(request.app.state, "EMBEDDING_LATENCY"):
        return {
            "embedding_latency": request.app.state.EMBEDDING_LATENCY,
            "search_latency": request.app.state.SEARCH_LATENCY
        }
    return None

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
        return {"modalities": modalities if modalities else ["text", "image", "audio"]}
    except Exception as e:
        logger.error(f"Error getting modalities: {e}")
        return {"modalities": ["text", "image", "audio"], "error": str(e)}

@router.get("/stats")
async def get_stats(
    storage_service = Depends(get_storage_service),
    performance_monitor = Depends(get_performance_monitor)
):
    try:
        if storage_service is None or storage_service.db is None:
            storage_stats = {
                "total_items": 0,
                "modality_counts": {"text": 0, "image": 0, "audio": 0, "video": 0},
                "category_counts": {},
                "status": "database_unavailable"
            }
        else:
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
            
            storage_stats = {
                "total_items": total_items,
                "modality_counts": modality_counts,
                "category_counts": category_counts,
                "status": "ok"
            }
     
        performance_stats = {}
        if performance_monitor:
            current_metrics = performance_monitor.collect_metrics()
            performance_stats = {
                "cpu_percent": current_metrics.get("cpu", {}).get("percent", 0),
                "memory_percent": current_metrics.get("memory", {}).get("percent", 0),
                "gpu": current_metrics.get("gpu", [])
            }
       
        service_stats = {}
        if hasattr(storage_service, 'get_stats'):
            service_stats = storage_service.get_stats()
        
        return {
            **storage_stats,
            "performance": performance_stats,
            "service": service_stats
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
    storage_service = Depends(get_storage_service),
    performance_monitor = Depends(get_performance_monitor),
    metrics = Depends(get_metrics)
):
    try:
        start_time = time.time()
       
        embedding_start = time.time()
        result = await embedding_service.process_text([query])
        query_embedding = result["embeddings"]
        embedding_time = time.time() - embedding_start
        
        if metrics and metrics["embedding_latency"]:
            metrics["embedding_latency"].labels(modality="text").observe(embedding_time)
      
        filters = {}
        if category:
            filters["category"] = category
        if modality:
            filters["modality"] = modality
       
        search_start = time.time()
        if filters:
            search_results = storage_service.filter_search(query_embedding, filters, k)
        else:
            search_results = storage_service.search(query_embedding, k)
        search_time = time.time() - search_start
        
        if metrics and metrics["search_latency"]:
            metrics["search_latency"].labels(modality="text").observe(search_time)
        
        total_time = time.time() - start_time
        if performance_monitor:
            performance_monitor.log_operation_metrics(
                operation="text_search",
                batch_size=1,
                latency=total_time,
                items_processed=len(search_results)
            )
        
        return {
            "query": query,
            "filters": filters,
            "results": search_results,
            "metrics": {
                "embedding_time": embedding_time,
                "search_time": search_time,
                "total_time": total_time,
                "num_results": len(search_results)
            }
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
    storage_service = Depends(get_storage_service),
    performance_monitor = Depends(get_performance_monitor),
    metrics = Depends(get_metrics)
):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}.jpg")
    
    try:
        start_time = time.time()
        
        with open(temp_path, "wb") as f:
            f.write(await file.read())
       
        embedding_start = time.time()
        result = await embedding_service.process_images([temp_path])
        query_embedding = result["embeddings"]
        embedding_time = time.time() - embedding_start
        
        if metrics and metrics["embedding_latency"]:
            metrics["embedding_latency"].labels(modality="image").observe(embedding_time)
        
        filters = {}
        if category:
            filters["category"] = category
        if modality:
            filters["modality"] = modality
        
        search_start = time.time()
        if filters:
            search_results = storage_service.filter_search(query_embedding, filters, k)
        else:
            search_results = storage_service.search(query_embedding, k)
        search_time = time.time() - search_start
        
        if metrics and metrics["search_latency"]:
            metrics["search_latency"].labels(modality="image").observe(search_time)
        
        total_time = time.time() - start_time
        if performance_monitor:
            performance_monitor.log_operation_metrics(
                operation="image_search",
                batch_size=1,
                latency=total_time,
                items_processed=len(search_results)
            )
        
        return {
            "filters": filters,
            "results": search_results,
            "metrics": {
                "embedding_time": embedding_time,
                "search_time": search_time,
                "total_time": total_time,
                "num_results": len(search_results)
            }
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
    storage_service = Depends(get_storage_service),
    performance_monitor = Depends(get_performance_monitor),
    metrics = Depends(get_metrics)
):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")
    
    try:
        start_time = time.time()
        
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        embedding_start = time.time()
        result = await embedding_service.process_audio([temp_path])
        query_embedding = result["embeddings"]
        embedding_time = time.time() - embedding_start
        
        if metrics and metrics["embedding_latency"]:
            metrics["embedding_latency"].labels(modality="audio").observe(embedding_time)
     
        filters = {"modality": "audio"}
        if category:
            filters["category"] = category
        
        search_start = time.time()
        search_results = storage_service.filter_search(query_embedding, filters, k)
        search_time = time.time() - search_start
       
        if metrics and metrics["search_latency"]:
            metrics["search_latency"].labels(modality="audio").observe(search_time)
       
        total_time = time.time() - start_time
        if performance_monitor:
            performance_monitor.log_operation_metrics(
                operation="audio_search",
                batch_size=1,
                latency=total_time,
                items_processed=len(search_results)
            )
        
        return {
            "filters": filters,
            "results": search_results,
            "metrics": {
                "embedding_time": embedding_time,
                "search_time": search_time,
                "total_time": total_time,
                "num_results": len(search_results)
            }
        }
    except Exception as e:
        logger.error(f"Error in audio search: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)