FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DEFAULT_TIMEOUT=300

# Install necessary system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    libsndfile1 \
    curl \
    wget \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch ecosystem packages separately to avoid timeouts
RUN pip install torch==1.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install compatible numpy
RUN pip install numpy==1.24.3

# Install required dependencies
RUN pip install \
    fastapi>=0.95.0 \
    uvicorn>=0.22.0 \
    faiss-cpu>=1.7.4 \
    pillow>=9.5.0 \
    python-multipart>=0.0.6 \
    prometheus-client>=0.17.0 \
    sqlalchemy>=2.0.0 \
    pymongo>=4.3.0 \
    pydantic>=2.0.0 \
    pandas \
    soundfile \
    ftfy \
    regex \
    einops \
    timm==0.6.7 \
    pytorchvideo==0.1.5

# Install ImageBind from GitHub
RUN git clone https://github.com/facebookresearch/ImageBind.git /tmp/imagebind && \
    cd /tmp/imagebind && \
    git checkout 3fcf5c9039de97f6ff5528ee4a9dce903c5979b3

# Create a custom setup.py that excludes problematic dependencies
RUN echo 'from setuptools import setup, find_packages\n\
setup(\n\
    name="imagebind",\n\
    version="0.1.0",\n\
    packages=find_packages(),\n\
    install_requires=[\n\
        "einops",\n\
        "ftfy",\n\
        "numpy>=1.19",\n\
        "regex",\n\
        "timm==0.6.7",\n\
        "torch==1.13.1",\n\
        "torchaudio",\n\
        "torchvision",\n\
        "pytorchvideo==0.1.5",\n\
    ],\n\
)' > /tmp/imagebind/setup.py

# Install ImageBind with modified setup.py
RUN cd /tmp/imagebind && pip install -e .

# Create directories for model caching and data
RUN mkdir -p /app/.checkpoints /app/data

# Create a special patch module for ImageBind
RUN echo 'import sys\n\
import types\n\
from unittest.mock import MagicMock\n\
\n\
def apply_patches():\n\
    """Apply patches to make ImageBind work without optional dependencies"""\n\
    # Create mock for cartopy\n\
    cartopy_mock = types.ModuleType("cartopy")\n\
    cartopy_mock.crs = MagicMock()\n\
    sys.modules["cartopy"] = cartopy_mock\n\
    \n\
    # Create mock for mayavi if needed\n\
    mayavi_mock = types.ModuleType("mayavi")\n\
    mayavi_mock.mlab = MagicMock()\n\
    sys.modules["mayavi"] = mayavi_mock\n\
    \n\
    # If eva_decord is imported but not available, mock it\n\
    if "eva_decord" not in sys.modules:\n\
        eva_mock = types.ModuleType("eva_decord")\n\
        sys.modules["eva_decord"] = eva_mock\n\
    \n\
    print("ImageBind patches applied successfully")\n\
\n\
# Apply patches when this module is imported\n\
apply_patches()' > /app/patch_imagebind.py

# Copy application code
COPY . .

# Add patch import to main.py if not already there
RUN if ! grep -q "import patch_imagebind" app/main.py; then \
    sed -i '1s/^/import sys\nimport os\nsys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\nimport patch_imagebind\n/' app/main.py; \
    fi

# Add a simple health check endpoint
RUN if ! grep -q "/health" app/main.py; then \
    echo '\n@app.get("/health")\ndef health():\n    return {"status": "healthy", "version": "1.0.0"}\n' >> app/main.py; \
    fi

# Expose port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]