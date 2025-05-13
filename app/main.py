import os
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
import time
import torch
from prometheus_client import Counter, Summary, Histogram, generate_latest, CONTENT_TYPE_LATEST

from app.api.routes import router as api_router
from app.storage.service import StorageService  # This can be the optimized version
from app.ingestion.service import RealIngestionService
from app.monitoring.performance import PerformanceMonitor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DB_URL = os.getenv('MONGODB_URL', 'mongodb://localhost:27017/multi_modal_search')

APP_HOST = os.getenv('APP_HOST', '0.0.0.0')
APP_PORT = int(os.getenv('APP_PORT', 8000))

# Initialize performance monitor
performance_monitor = PerformanceMonitor(log_dir="logs/performance", interval=10.0)

# Create app instance
app = FastAPI(title="Multi-Modal Search API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP Requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Summary("http_request_duration_seconds", "HTTP Request Latency", ["method", "endpoint"])
EMBEDDING_LATENCY = Histogram("embedding_generation_seconds", "Embedding Generation Latency", ["modality"])
SEARCH_LATENCY = Histogram("search_operation_seconds", "Search Operation Latency", ["modality"])

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    # Record request metrics
    duration = time.time() - start_time
    REQUEST_COUNT.labels(request.method, request.url.path, response.status_code).inc()
    REQUEST_LATENCY.labels(request.method, request.url.path).observe(duration)
    
    # Log to performance monitor
    performance_monitor.log_operation_metrics(
        operation=f"{request.method}_{request.url.path}",
        batch_size=1,
        latency=duration,
        items_processed=1
    )
    
    return response

# Include API routes
app.include_router(api_router, prefix="/api")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "version": "1.0.0"}

@app.get("/performance")
async def performance_stats():
    """Get current performance statistics"""
    current_metrics = performance_monitor.collect_metrics()
    storage_stats = {}
    
    if hasattr(app.state, 'storage_service') is not None:
        storage_stats = app.state.storage_service.get_stats()
    
    return {
        "system": current_metrics,
        "storage": storage_stats
    }

# Mount static files for UI
app.mount("/static", StaticFiles(directory="app/ui"), name="static")
app.mount("/data", StaticFiles(directory="data"), name="data")

# Serve UI
@app.get("/")
async def serve_ui():
    return FileResponse("app/ui/index.html")

# Initialize data and services on startup
@app.on_event("startup")
async def startup_event():
    try:
        # Start performance monitoring
        performance_monitor.start_monitoring()
        logger.info("Performance monitoring started")
        
        # Initialize directories
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/indices", exist_ok=True)
        os.makedirs("temp", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("logs/performance", exist_ok=True)
        
        # Initialize and attach storage service to app state
        logger.info("Initializing storage service...")
        storage_service = StorageService(mongo_uri=DB_URL, use_mock_if_unavailable=True)
        app.state.storage_service = storage_service
        
        # Attach performance monitor to app state
        app.state.performance_monitor = performance_monitor
        
        # Log storage service stats
        storage_stats = storage_service.get_stats()
        logger.info(f"Storage service initialized: {storage_stats}")
        
        try:
            if storage_service.db is not None:
                metadata_count = storage_service.db.metadata.count_documents({})
                logger.info(f"Current metadata documents: {metadata_count}")
                
                if metadata_count == 0:
                    logger.info("No data found. Starting data ingestion process...")
                    ingestion_service = RealIngestionService()
                    
                    # Data ingestion will be performed as a background task
                    ingestion_result = await ingestion_service.ingest_batch()
                    logger.info(f"Data ingestion completed: {ingestion_result}")
                else:
                    logger.info(f"Found {metadata_count} existing metadata documents")
            else:
                logger.warning("Database not available, skipping data ingestion")
        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
      
        logger.info(f"PyTorch version: {torch.__version__}")
        gpu_available = torch.cuda.is_available()
        
        if gpu_available:
            try:
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"GPU available: {gpu_count} device(s), Primary: {gpu_name}")
            except Exception as e:
                logger.warning(f"GPU detection error: {e}. Running in CPU mode.")
        else:
            logger.info("No GPU available, using CPU")
        
        logger.info("Startup complete")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    try:
        # Stop performance monitoring
        performance_monitor.stop_monitoring()
        logger.info("Performance monitoring stopped")
        
        # Save any pending performance data
        performance_monitor.save_performance_data()
        
        # Get final stats
        if hasattr(app.state, 'storage_service') is not None:
            final_stats = app.state.storage_service.get_stats()
            logger.info(f"Final storage stats: {final_stats}")
        
        logger.info("Shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Export metrics for external use
app.state.EMBEDDING_LATENCY = EMBEDDING_LATENCY
app.state.SEARCH_LATENCY = SEARCH_LATENCY

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT, reload=True)