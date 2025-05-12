import torch
from typing import Dict, Any, List, Optional
import asyncio
import time
import threading
from queue import Queue

class AsyncInferenceOptimizer:
    """
    Optimizes inference by using asynchronous processing and batching
    """
    def __init__(self, model, default_batch_size: int = 32, max_queue_size: int = 100):
        self.model = model
        self.default_batch_size = default_batch_size
        self.processing_queue = Queue(maxsize=max_queue_size)
        self.results = {}
        self.stop_event = threading.Event()
        self.worker_thread = None
    
    def start_worker(self):
        """Start the background worker thread"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.stop_event.clear()
            self.worker_thread = threading.Thread(target=self._process_queue)
            self.worker_thread.daemon = True
            self.worker_thread.start()
    
    def stop_worker(self):
        """Stop the background worker thread"""
        self.stop_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
    
    def _process_queue(self):
        """Background worker that processes batches"""
        batch = []
        batch_ids = []
        last_process_time = time.time()
        
        while not self.stop_event.is_set():
            # Process if we have a full batch or timeout occurred
            current_time = time.time()
            timeout_occurred = current_time - last_process_time > 0.1  # 100ms timeout
            
            if len(batch) >= self.default_batch_size or (timeout_occurred and batch):
                # Process batch
                with torch.no_grad():
                    # Perform inference
                    outputs = self.model(batch)
                
                # Store results
                for batch_id, output in zip(batch_ids, outputs):
                    self.results[batch_id] = output
                
                # Reset batch
                batch = []
                batch_ids = []
                last_process_time = current_time
            
            # Get item from queue (non-blocking)
            try:
                item_id, item = self.processing_queue.get(block=False)
                batch.append(item)
                batch_ids.append(item_id)
            except:
                # Queue empty, sleep briefly
                time.sleep(0.01)  # 10ms
        
        # Process any remaining items before stopping
        if batch:
            with torch.no_grad():
                outputs = self.model(batch)
            
            for batch_id, output in zip(batch_ids, outputs):
                self.results[batch_id] = output
    
    async def async_inference(self, input_data: Any, item_id: Optional[str] = None) -> Any:
        """
        Submit inference job and wait for result asynchronously
        
        Args:
            input_data: Input data for inference
            item_id: Optional ID for the inference job
        
        Returns:
            Inference result
        """
        # Generate random ID if not provided
        if item_id is None:
            item_id = str(time.time())
        
        # Make sure worker is running
        self.start_worker()
        
        # Add to processing queue
        self.processing_queue.put((item_id, input_data))
        
        # Wait for result (with timeout)
        max_wait_time = 30.0  # 30 seconds timeout
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            if item_id in self.results:
                result = self.results.pop(item_id)
                return result
            
            # Yield to allow other async operations
            await asyncio.sleep(0.01)  # 10ms
        
        # Timeout occurred
        raise TimeoutError(f"Inference timeout after {max_wait_time} seconds")