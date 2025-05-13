
import os
import logging
import time
from typing import List
from tqdm import tqdm
from app.storage.models import MetadataModel
from app.embedding.service import EmbeddingService
from app.storage.service import StorageService
from app.ingestion.loader import RealDataLoader
import gc  
import torch 

logger = logging.getLogger(__name__)

# Optimized batch sizes to prevent memory issues
IMAGE_SAMPLES = 50
AUDIO_SAMPLES = 50
TEXT_SAMPLES = 100
VIDEO_SAMPLES = 0   # Due to size constraint

# Memory-friendly batch sizes
EMBEDDING_BATCH_SIZE = 8  # Reduce if still getting memory issues
STORAGE_BATCH_SIZE = 16

class RealIngestionService:
    def __init__(self):
        self.data_loader = RealDataLoader()
        self.embedding_service = EmbeddingService()
        self.storage_service = StorageService(use_mock_if_unavailable=False)
    
    async def ingest_batch(self, batch_size=EMBEDDING_BATCH_SIZE):
        """Optimized ingestion with memory management"""
        try:
            if not hasattr(self.storage_service, 'db') or self.storage_service.db is None:
                logger.error("MongoDB not available")
                return {"status": "error", "message": "MongoDB not available"}
            
            print("Starting real data ingestion...")
            start_time = time.time()

            # Download data
            all_metadata = self.data_loader.download_all_data(
                image_samples=IMAGE_SAMPLES,
                audio_samples=AUDIO_SAMPLES,
                text_samples=TEXT_SAMPLES,
                video_samples=VIDEO_SAMPLES
            )
            
            # Group by modality
            modality_groups = {}
            for metadata in all_metadata:
                if metadata.modality not in modality_groups:
                    modality_groups[metadata.modality] = []
                modality_groups[metadata.modality].append(metadata)
        
            results = {}
            
            # Process each modality with memory management
            for modality, items in modality_groups.items():
                if modality == "text":
                    print(f"\nProcessing {len(items)} text items...")
                    results["text"] = await self._process_modality_with_memory_management(
                        items, self._process_text_batch, batch_size
                    )
                elif modality == "image":
                    print(f"\nProcessing {len(items)} image items...")
                    results["image"] = await self._process_modality_with_memory_management(
                        items, self._process_image_batch, batch_size
                    )
                elif modality == "audio":
                    print(f"\nProcessing {len(items)} audio items...")
                    results["audio"] = await self._process_modality_with_memory_management(
                        items, self._process_audio_batch, batch_size
                    )
                
                # Clean up memory after each modality
                self._cleanup_memory()
            
            total_time = time.time() - start_time
            
            # Calculate total processed items
            total_processed = sum(
                sum(result.get("processed", 0) for result in results.get(modality, []))
                for modality in results
            )
            
            print(f"\nIngestion completed: {total_processed} items in {total_time:.2f} seconds")
            
            return {
                "status": "success",
                "total_processed": total_processed,
                "total_time": total_time,
                "average_throughput": total_processed / total_time if total_time > 0 else 0,
                "results_by_modality": results
            }
            
        except Exception as e:
            logger.error(f"Error in ingest_batch: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": f"Error during ingestion: {str(e)}"}
    
    async def _process_modality_with_memory_management(self, items, process_func, batch_size):
        """Process items with memory management to prevent crashes"""
        results = []
        
        for i in tqdm(range(0, len(items), batch_size)):
            batch = items[i:i+batch_size]
            
            # Process batch
            result = await process_func(batch)
            results.append(result)
            
            # Print progress
            processed = i + len(batch)
            print(f"Progress: {processed}/{len(items)} ({processed/len(items)*100:.1f}%)")
            
            # Clean up memory periodically
            if i % (batch_size * 4) == 0 and i > 0:
                self._cleanup_memory()
                time.sleep(0.5)  # Give system time to release memory
        
        return results
    
    def _cleanup_memory(self):
        """Clean up memory to prevent device crashes"""
        # Force garbage collection
        gc.collect()
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("Memory cleanup completed")
    
    async def _process_text_batch(self, metadata_batch: List[MetadataModel]):
        """Process text batch with better error handling"""
        try:
            texts = []
            valid_metadata = []
            
            for metadata in metadata_batch:
                try:
                    with open(metadata.source_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        # Truncate very long texts to prevent memory issues
                        if len(text) > 5000:
                            text = text[:5000]
                    texts.append(text)
                    valid_metadata.append(metadata)
                except Exception as e:
                    logger.error(f"Error reading text file {metadata.source_path}: {e}")
                    # Use metadata description as fallback
                    texts.append(f"{metadata.title}. {metadata.description}")
                    valid_metadata.append(metadata)
            
            if not texts:
                return {"processed": 0}
            
            # Generate embeddings
            result = await self.embedding_service.process_text(texts)
            embeddings = result["embeddings"]
            
            # Store in batches
            return await self._store_embeddings_batch(valid_metadata, embeddings, "text")
            
        except Exception as e:
            logger.error(f"Error in process_text_batch: {e}")
            return {"processed": 0, "error": str(e)}
    
    async def _process_image_batch(self, metadata_batch: List[MetadataModel]):
        """Process image batch with memory optimization"""
        try:
            valid_metadata = []
            image_paths = []
            
            for metadata in metadata_batch:
                if os.path.exists(metadata.source_path):
                    valid_metadata.append(metadata)
                    image_paths.append(metadata.source_path)
            
            if not image_paths:
                return {"processed": 0}
            
            # Generate embeddings
            result = await self.embedding_service.process_images(image_paths)
            embeddings = result["embeddings"]
            
            # Store in batches
            return await self._store_embeddings_batch(valid_metadata, embeddings, "image")
            
        except Exception as e:
            logger.error(f"Error in process_image_batch: {e}")
            return {"processed": 0, "error": str(e)}
    
    async def _process_audio_batch(self, metadata_batch: List[MetadataModel]):
        """Process audio batch with memory optimization"""
        try:
            valid_metadata = []
            audio_paths = []
            
            for metadata in metadata_batch:
                if os.path.exists(metadata.source_path):
                    valid_metadata.append(metadata)
                    audio_paths.append(metadata.source_path)
            
            if not audio_paths:
                return {"processed": 0}
            
            # Generate embeddings
            result = await self.embedding_service.process_audio(audio_paths)
            embeddings = result["embeddings"]
            
            # Store in batches
            return await self._store_embeddings_batch(valid_metadata, embeddings, "audio")
            
        except Exception as e:
            logger.error(f"Error in process_audio_batch: {e}")
            return {"processed": 0, "error": str(e)}
    
    async def _store_embeddings_batch(self, metadata_list, embeddings, modality):
        """Store embeddings in smaller batches to prevent memory issues"""
        total_processed = 0
        
        # Process in smaller storage batches
        for i in range(0, len(metadata_list), STORAGE_BATCH_SIZE):
            batch_metadata = metadata_list[i:i+STORAGE_BATCH_SIZE]
            batch_embeddings = embeddings[i:i+STORAGE_BATCH_SIZE]
            
            metadata_ids = []
            item_ids = []
            
            # Store metadata
            for metadata in batch_metadata:
                try:
                    metadata_id = self.storage_service.store_metadata(metadata)
                    metadata_ids.append(metadata_id)
                    item_ids.append(metadata.id)
                except Exception as e:
                    logger.error(f"Error storing metadata: {e}")
            
            # Store embeddings
            if item_ids and metadata_ids:
                try:
                    self.storage_service.batch_store_embeddings(
                        item_ids, metadata_ids, batch_embeddings, modality
                    )
                    total_processed += len(item_ids)
                except Exception as e:
                    logger.error(f"Error storing embeddings: {e}")
            
            # Save index periodically
            if i % (STORAGE_BATCH_SIZE * 4) == 0:
                self.storage_service.save_index()
        
        return {"processed": total_processed}