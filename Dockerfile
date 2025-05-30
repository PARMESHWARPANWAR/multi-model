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

# Install Python packages in multiple steps with pip optimization flags
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir wheel setuptools && \
    pip install --no-cache-dir torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
RUN pip install --no-cache-dir -r requirements.txt

# Install pytorchvideo
RUN pip install --no-cache-dir pytorchvideo

# Install ImageBind without dependencies
RUN pip install --no-deps git+https://github.com/facebookresearch/ImageBind.git

# Copy the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV KMP_DUPLICATE_LIB_OK=TRUE
# Atlas MongoDB URL will be passed via docker-compose or environment
ENV MONGODB_URL=mongodb+srv://Ecom-Mern:orzNcirHTxouOmba@cluster0.2j1hm.mongodb.net/multi_modal_search?retryWrites=true&w=majority

# Create required directories
RUN mkdir -p data/raw data/indices temp logs

# Expose the port
EXPOSE 8000

COPY download_model.py .
RUN python download_model.py

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]