import torch
import numpy as np
import logging
import gc
import os
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self, memory_efficient=True):
        # Set memory allocation strategy
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        # Clear any existing GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Memory efficient loading
        if memory_efficient and self.device == "cuda":
            # Load model to CPU first
            self.model = imagebind_model.imagebind_huge(pretrained=True)
            self.model.eval()
            
            # Move to GPU with memory efficiency
            self.model = self.model.half()  # Use float16 to save memory
            self.model.to(self.device)
        else:
            self.model = imagebind_model.imagebind_huge(pretrained=True)
            self.model.eval()
            self.model.to(self.device)
        
        self.use_amp = torch.cuda.is_available()
        
        # Reduced batch sizes for memory efficiency
        self.batch_sizes = {
            ModalityType.TEXT: 128,  # Reduced from 256
            ModalityType.VISION: 32,  # Reduced from 64
            ModalityType.AUDIO: 16,   # Reduced from 32
        }
        
        # Clear cache after initialization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def generate_embedding(self, modality_type, inputs, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_sizes.get(modality_type, 16)
        
        try:
            if torch.is_tensor(inputs):
                with torch.no_grad():
                    # Clear cache before processing
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            embeddings = self.model({modality_type: inputs})
                    else:
                        embeddings = self.model({modality_type: inputs})
                    
                    result = embeddings[modality_type].cpu().numpy()
                    
                    # Clear GPU memory immediately
                    del embeddings
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    
                    return result
            
            # Batch processing for lists
            all_embeddings = []
            
            with torch.no_grad():
                for i in range(0, len(inputs), batch_size):
                    batch = inputs[i:i+batch_size]
                    inputs_dict = {modality_type: batch}
                    
                    # Clear cache before each batch
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            embeddings = self.model(inputs_dict)
                    else:
                        embeddings = self.model(inputs_dict)
                    
                    embeddings_np = embeddings[modality_type].cpu().numpy()
                    all_embeddings.append(embeddings_np)
                    
                    # Aggressive memory cleanup
                    del embeddings
                    del inputs_dict
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
            
            return np.concatenate(all_embeddings, axis=0) if len(all_embeddings) > 1 else all_embeddings[0]
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"GPU OOM with batch_size={batch_size}, retrying with smaller batch")
                torch.cuda.empty_cache()
                # Retry with half the batch size
                return self.generate_embedding(modality_type, inputs, batch_size=batch_size//2)
            else:
                raise e
    
    
    def __del__(self):
        # Cleanup when object is destroyed
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()        

