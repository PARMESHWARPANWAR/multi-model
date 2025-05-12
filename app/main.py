import os
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
import time
from prometheus_client import Counter, Summary, Histogram, generate_latest, CONTENT_TYPE_LATEST

from app.api.routes import router as api_router
from app.storage.service import StorageService
from app.ingestion.real_loader import RealIngestionService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_URL = 'mongodb://localhost:27017/'

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
    
    return response

app.include_router(api_router, prefix="/api")

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}

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
        # Initialize directories
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/indices", exist_ok=True)
        os.makedirs("temp", exist_ok=True)
        
        # Initialize and attach storage service to app state
        logger.info("Initializing storage service...")
        storage_service = StorageService(mongo_uri=DB_URL, use_mock_if_unavailable=True)
        app.state.storage_service = storage_service
        
        # Attempt to initialize data if needed
        try:
            if storage_service.db is not None:
                metadata_count = storage_service.db.metadata.count_documents({})
                if metadata_count == 0:
                    logger.info("No data found. Starting data ingestion process...")
                    ingestion_service = RealIngestionService()
                    # Data ingestion will be performed as a background task
                    ingestion_result = await ingestion_service.ingest_batch()
                    logger.info(f"Data ingestion completed: {ingestion_result}")
            else:
                logger.warning("Database not available, skipping data ingestion")
        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
        
        logger.info("Startup complete")
    except Exception as e:
        logger.error(f"Error during startup: {e}")