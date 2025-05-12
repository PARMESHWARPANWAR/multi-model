# Multi-Modal Embedding & Search API

A highly optimized system for multi-modal (text, image, audio) embedding generation and similarity search, built to maximize throughput on GPU hardware.

## Architecture Overview

This system features a microservices architecture designed to efficiently generate, store, and search multi-modal embeddings at scale.

![Architecture Diagram](https://via.placeholder.com/800x500)

### Key Components

1. **Embedding Service**: GPU-accelerated service for generating embeddings across multiple modalities
2. **Storage Service**: Optimized vector database + metadata storage
3. **API Service**: Handles search and ingestion requests
4. **Ingestion Pipeline**: Supports both batch and streaming data ingestion
5. **Monitoring Service**: Collects and visualizes performance metrics

## Performance Optimizations

The system implements several key performance optimizations to maximize throughput on GPU hardware:

### Embedding Generation Optimizations

- **Dynamic Batching**: Automatically adjusts batch sizes to maximize GPU utilization
- **Mixed Precision**: Uses FP16 for faster computation where appropriate
- **GPU Memory Management**: Optimized memory usage patterns to prevent OOM errors
- **Asynchronous Processing**: Non-blocking operations for maximum throughput
- **Embedding Quantization**: Optional 8-bit quantization for reduced memory footprint

### Vector Search Optimizations

- **FAISS GPU Acceleration**: GPU-accelerated similarity search
- **Index Optimization**: Custom FAISS index configuration for maximum performance
- **Metadata Caching**: In-memory cache for frequently accessed metadata
- **Parallel Query Processing**: Process multiple queries simultaneously
- **Query Optimization**: Efficient query planning for filtered searches

### System-Level Optimizations

- **Async I/O**: Non-blocking I/O operations throughout the system
- **Resource Monitoring**: Real-time monitoring of GPU, CPU, and memory usage
- **Adaptive Scaling**: Dynamic resource allocation based on workload
- **Request Batching**: Aggregates requests for more efficient processing

## Getting Started

### Prerequisites

- NVIDIA GPU with CUDA support (A100 recommended)
- Docker and Docker Compose
- Python 3.8+

### Installation

1. Clone the repository:

```bash
git clone https://github.com/PARMESHWARPANWAR/multimodal-search-api.git
cd multimodal-search-api
```

2. Build and start the services:

```bash
docker-compose up -d
```

3. Initialize the database (first time only):

```bash
docker-compose exec app python -m app.scripts.initialize_db
```

### Sample Usage

#### Searching with Text

```bash
curl -X POST "http://localhost:8000/api/search/text" \
     -H "Content-Type: application/json" \
     -d '{"query": "a beautiful sunset over mountains", "k": 10}'
```

#### Searching with an Image

```bash
curl -X POST "http://localhost:8000/api/search/image" \
     -F "file=@/path/to/your/image.jpg" \
     -F "k=10"
```

#### Searching with Audio

```bash
curl -X POST "http://localhost:8000/api/search/audio" \
     -F "file=@/path/to/your/audio.wav" \
     -F "k=10"
```

## Performance Benchmarks

The system has been benchmarked on an NVIDIA A100 GPU with the following results:

| Metric | Value |
|--------|-------|
| Embedding Generation Throughput | ~500 images/second |
| Vector Search Throughput | ~10,000 queries/second |
| End-to-End Latency (P95) | 150ms |
| Maximum Batch Size (Images) | 64 |
| Maximum Batch Size (Text) | 256 |
| GPU Memory Usage | 16GB |

## System Monitoring

The system includes comprehensive monitoring via Prometheus and Grafana:

- **Access Grafana**: Open http://localhost:3000 (default credentials: admin/admin)
- **Default Dashboard**: "Multi-Modal Search API Dashboard"

Monitored metrics include:
- Embedding generation throughput and latency
- Search operation throughput and latency
- GPU utilization and memory usage
- API request rates and error rates

## Project Structure

```
multimodal-search-api/
├── app/
│   ├── api/              # API endpoints
│   ├── embedding/        # Embedding generation service
│   ├── ingestion/        # Data ingestion pipeline
│   ├── monitoring/       # Performance monitoring
│   ├── optimizations/    # Performance optimizations
│   ├── storage/          # Vector and metadata storage
│   └── ui/               # Web interface
├── data/
│   ├── indices/          # FAISS indices
│   └── raw/              # Raw data files
├── logs/                 # Log files
├── monitoring/
│   ├── grafana/          # Grafana dashboards
│   └── prometheus/       # Prometheus configuration
├── docker-compose.yml    # Service configuration
├── Dockerfile            # Container definition
└── requirements.txt      # Python dependencies
```

## Advanced Configuration

### Tuning Batch Sizes

Adjust batch sizes in `app/embedding/service.py`:

```python
self.optimal_batch_sizes = {
    ModalityType.TEXT: 256,  # Increase for more powerful GPUs
    ModalityType.VISION: 64,
    ModalityType.AUDIO: 32,
}
```

### FAISS Index Configuration

Modify index configuration in `app/storage/faiss_gpu.py`:

```python
# For higher recall at the expense of query speed
self.cpu_index = faiss.IndexFlatIP(dimension)

# For faster queries at the expense of recall
# self.cpu_index = faiss.IndexIVFFlat(quantizer, dimension, num_clusters)
```

### Memory Optimization

Adjust cache sizes in `app/storage/service.py`:

```python
# Increase for systems with more RAM
self.cache_size = 10000
```

## Extending the System

### Adding a New Modality

1. Implement data loading in `app/ingestion/data_loader.py`
2. Add embedding generation in `app/embedding/service.py`
3. Add API endpoint in `app/api/routes.py`
4. Update UI to support the new modality

### Using a Different Embedding Model

To replace ImageBind with another model:

1. Update the model implementation in `app/embedding/model.py`
2. Adjust preprocessing in the embedding service
3. Update vector dimensions in the storage service

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Facebook Research for the ImageBind model
- FAISS team for the efficient similarity search library