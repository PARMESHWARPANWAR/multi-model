import time
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any
from imagebind.models.imagebind_model import ModalityType
from imagebind import data
from app.embedding.model import EmbeddingModel

import logging
logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.model = EmbeddingModel()
        self.device = self.model.device

        self.use_mixed_precision = torch.cuda.is_available()
        self.use_gpu_optimization = torch.cuda.is_available()
      
        if self.use_gpu_optimization:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            logger.info("GPU optimizations enabled")
       
        self.total_requests = 0
        self.total_processing_time = 0
    
    async def process_text(self, texts: List[str]) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            inputs = data.load_and_transform_text(texts, self.device)
            
            with torch.no_grad():
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        embeddings = self.model.generate_embedding(ModalityType.TEXT, inputs)
                else:
                    embeddings = self.model.generate_embedding(ModalityType.TEXT, inputs)
            
            elapsed_time = time.time() - start_time
            self.total_requests += len(texts)
            self.total_processing_time += elapsed_time
            
            logger.info(f"Processed {len(texts)} texts in {elapsed_time:.3f}s")
            
            return {
                "embeddings": embeddings,
                "metrics": {
                    "latency": elapsed_time,
                    "throughput": len(texts) / elapsed_time,
                    "total_requests": self.total_requests,
                    "avg_latency": self.total_processing_time / self.total_requests if self.total_requests > 0 else 0
                }
            }
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return {
                "embeddings": np.zeros((len(texts), 1024)),
                "metrics": {
                    "latency": time.time() - start_time,
                    "error": str(e)
                }
            }
    
    async def process_images(self, image_paths: List[str]) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            inputs = data.load_and_transform_vision_data(image_paths, self.device)
          
            with torch.no_grad():
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        embeddings = self.model.generate_embedding(ModalityType.VISION, inputs)
                else:
                    embeddings = self.model.generate_embedding(ModalityType.VISION, inputs)
            
            # Compute metrics
            elapsed_time = time.time() - start_time
            self.total_requests += len(image_paths)
            self.total_processing_time += elapsed_time
            
            logger.info(f"Processed {len(image_paths)} images in {elapsed_time:.3f}s")
            
            return {
                "embeddings": embeddings,
                "metrics": {
                    "latency": elapsed_time,
                    "throughput": len(image_paths) / elapsed_time,
                    "total_requests": self.total_requests,
                    "avg_latency": self.total_processing_time / self.total_requests if self.total_requests > 0 else 0
                }
            }
        except Exception as e:
            logger.error(f"Error processing images: {e}")
            return {
                "embeddings": np.zeros((len(image_paths), 1024)),
                "metrics": {
                    "latency": time.time() - start_time,
                    "error": str(e)
                }
            }
    
    async def process_audio(self, audio_paths: List[str]) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            inputs = data.load_and_transform_audio_data(audio_paths, self.device)
      
            with torch.no_grad():
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        embeddings = self.model.generate_embedding(ModalityType.AUDIO, inputs)
                else:
                    embeddings = self.model.generate_embedding(ModalityType.AUDIO, inputs)
            
            elapsed_time = time.time() - start_time
            self.total_requests += len(audio_paths)
            self.total_processing_time += elapsed_time
            
            logger.info(f"Processed {len(audio_paths)} audio files in {elapsed_time:.3f}s")
            
            return {
                "embeddings": embeddings,
                "metrics": {
                    "latency": elapsed_time,
                    "throughput": len(audio_paths) / elapsed_time,
                    "total_requests": self.total_requests,
                    "avg_latency": self.total_processing_time / self.total_requests if self.total_requests > 0 else 0
                }
            }
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return {
                "embeddings": np.zeros((len(audio_paths), 1024)),
                "metrics": {
                    "latency": time.time() - start_time,
                    "error": str(e)
                }
            }
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "total_processing_time": self.total_processing_time,
            "avg_latency": self.total_processing_time / self.total_requests if self.total_requests > 0 else 0,
            "device": str(self.device),
            "mixed_precision_enabled": self.use_mixed_precision,
            "gpu_optimization_enabled": self.use_gpu_optimization
        }