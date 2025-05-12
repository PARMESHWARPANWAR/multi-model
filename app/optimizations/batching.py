import torch
import numpy as np
from typing import List, Dict, Any, Callable, TypeVar, Generic

T = TypeVar('T')

class DynamicBatcher(Generic[T]):
    """
    Implements dynamic batching to optimize GPU utilization
    """
    def __init__(self, 
                 processing_fn: Callable[[List[T]], Any],
                 max_batch_size: int = 32,
                 timeout: float = 0.1):
        """
        Args:
            processing_fn: Function that processes a batch of items
            max_batch_size: Maximum batch size
            timeout: Maximum time to wait for batch to fill up (seconds)
        """
        self.processing_fn = processing_fn
        self.max_batch_size = max_batch_size
        self.timeout = timeout
        self.queue = []
        self.processing = False
    
    async def add_item(self, item: T) -> Any:
        """
        Add an item to the batch queue and process when ready
        """
        self.queue.append(item)
        
        # If we've hit max batch size, process immediately
        if len(self.queue) >= self.max_batch_size:
            return await self.process_batch()
        
        # Otherwise, start a timer if not already processing
        if not self.processing:
            self.processing = True
            # Start async timer
            # In a real implementation, this would use asyncio.create_task
            # For this example, we'll process immediately
            return await self.process_batch()
    
    async def process_batch(self) -> List[Any]:
        """
        Process the current batch of items
        """
        if not self.queue:
            self.processing = False
            return []
        
        # Get current items in queue
        batch = self.queue.copy()
        self.queue = []
        
        # Process the batch
        results = await self.processing_fn(batch)
        
        self.processing = False
        return results


class GPUOptimizer:
    """
    Optimize GPU memory usage and computation
    """
    def __init__(self):
        self.fp16_enabled = torch.cuda.is_available()
    
    def enable_mixed_precision(self):
        """
        Enable mixed precision training (FP16)
        """
        if not self.fp16_enabled:
            return False
        
        # In a real implementation, we would use torch.cuda.amp
        # For this example, we'll just simulate it
        print("Mixed precision enabled")
        return True
    
    def optimize_memory_usage(self, model):
        """
        Apply memory optimizations to model
        """
        if not torch.cuda.is_available():
            return
        
        # Free unused memory
        torch.cuda.empty_cache()
        
        # In a real implementation, we might apply gradient checkpointing or other techniques
        print(f"GPU memory optimizations applied. Available memory: {torch.cuda.get_device_properties(0).total_memory}")


class EmbeddingOptimizer:
    """
    Optimize embedding generation and storage
    """
    def __init__(self, embedding_dim=1024):
        self.embedding_dim = embedding_dim
        
    def quantize_embeddings(self, embeddings: np.ndarray, bits: int = 8) -> np.ndarray:
        """
        Quantize embeddings to reduce memory footprint
        
        Args:
            embeddings: Embeddings to quantize
            bits: Number of bits to quantize to (8 or 16)
        
        Returns:
            Quantized embeddings
        """
        if bits not in [8, 16]:
            raise ValueError("Bits must be either 8 or 16")
        
        # Determine min and max values
        min_val = embeddings.min()
        max_val = embeddings.max()
        
        # Calculate scale and zero point
        scale = (max_val - min_val) / (2**bits - 1)
        zero_point = -min_val / scale
        
        # Quantize
        if bits == 8:
            quantized = np.round(embeddings / scale + zero_point).astype(np.uint8)
        else:  # bits == 16
            quantized = np.round(embeddings / scale + zero_point).astype(np.uint16)
        
        # Store quantization parameters
        quantization_params = {
            'min_val': min_val,
            'max_val': max_val,
            'scale': scale,
            'zero_point': zero_point,
            'bits': bits
        }
        
        return quantized, quantization_params
    
    def dequantize_embeddings(self, quantized_embeddings: np.ndarray, 
                             quantization_params: Dict[str, Any]) -> np.ndarray:
        """
        Dequantize embeddings back to original precision
        
        Args:
            quantized_embeddings: Quantized embeddings
            quantization_params: Parameters used for quantization
        
        Returns:
            Dequantized embeddings
        """
        scale = quantization_params['scale']
        zero_point = quantization_params['zero_point']
        
        # Dequantize
        dequantized = (quantized_embeddings.astype(np.float32) - zero_point) * scale
        
        return dequantized