import time
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any
from imagebind.models.imagebind_model import ModalityType
from imagebind import data
from app.embedding.model import EmbeddingModel

class EmbeddingService:
    def __init__(self):
        self.model = EmbeddingModel()
        
    async def process_text(self, texts: List[str]) -> Dict[str, Any]:
        """Process text inputs and return embeddings"""
        start_time = time.time()
        
        # Prepare inputs
        inputs = data.load_and_transform_text(texts, self.model.device)
        
        # Generate embeddings
        embeddings = self.model.generate_embedding(ModalityType.TEXT, inputs)
        
        # Compute metrics
        elapsed_time = time.time() - start_time
        
        return {
            "embeddings": embeddings,
            "metrics": {
                "latency": elapsed_time,
                "throughput": len(texts) / elapsed_time
            }
        }
    
    async def process_images(self, image_paths: List[str]) -> Dict[str, Any]:
        """Process image inputs and return embeddings"""
        start_time = time.time()
        
        # Prepare inputs
        inputs = data.load_and_transform_vision_data(image_paths, self.model.device)
        
        # Generate embeddings
        embeddings = self.model.generate_embedding(ModalityType.VISION, inputs)
        
        # Compute metrics
        elapsed_time = time.time() - start_time
        
        return {
            "embeddings": embeddings,
            "metrics": {
                "latency": elapsed_time,
                "throughput": len(image_paths) / elapsed_time
            }
        }
    
    async def process_audio(self, audio_paths: List[str]) -> Dict[str, Any]:
        """Process audio inputs and return embeddings"""
        start_time = time.time()
        
        # Prepare inputs
        inputs = data.load_and_transform_audio_data(audio_paths, self.model.device)
        
        # Generate embeddings
        embeddings = self.model.generate_embedding(ModalityType.AUDIO, inputs)
        
        # Compute metrics
        elapsed_time = time.time() - start_time
        
        return {
            "embeddings": embeddings,
            "metrics": {
                "latency": elapsed_time,
                "throughput": len(audio_paths) / elapsed_time
            }
        }