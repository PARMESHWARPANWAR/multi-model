FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install pytorchvideo
RUN pip install pytorchvideo

# Install ImageBind without dependencies
RUN pip install --no-deps git+https://github.com/facebookresearch/ImageBind.git

# Copy the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV MONGODB_URL=mongodb://mongodb:27017/multi_modal_search

# Create required directories
RUN mkdir -p data/raw data/indices temp logs

# Expose the port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]