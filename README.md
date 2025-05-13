# Multi-Modal Search System

Search across text, images, and audio using ImageBind embeddings.

## Requirements
- Python 3.10.9
- MongoDB
- Git

## Installation

### Option 1: Local Setup

```bash
# Clone repository
git clone https://github.com/yourusername/multi-modal-search.git
cd multi-modal-search

# Create virtual environment
python -m venv multi_model_env

# Activate virtual environment
# Windows:
multi_model_env\Scripts\activate
# Linux/Mac:
source multi_model_env/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install --no-deps git+https://github.com/facebookresearch/ImageBind.git
pip install pytorchvideo

# Start MongoDB (if not running)
docker run -d -p 27017:27017 --name mongodb mongo:latest

# Run application
uvicorn app.main:app --reload
```
### Option 2: Docker

```bash
# Build and run
docker-compose build
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
