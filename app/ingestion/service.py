import os
import time
import logging
import asyncio
import uuid
import random
from typing import List, Dict, Any
import numpy as np
import torch
from app.storage.models import MetadataModel
from app.embedding.service import EmbeddingService
from app.storage.service import StorageService

logger = logging.getLogger(__name__)

class DataLoader:
    """Simulated data loader for demo purposes"""
    
    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Create subdirectories for each modality
        for modality in ["text", "image", "audio"]:
            os.makedirs(os.path.join(data_dir, modality), exist_ok=True)
    
    def download_all_data(self, sample_mode=True) -> List[MetadataModel]:
        if sample_mode:
            return self._generate_sample_metadata()
    
        logger.info("Downloading 5000 images from ImageNet subset")
        logger.info("Downloading 2000 audio clips from AudioSet")
        logger.info("Downloading 3000 text samples from IMDB")
        
        return self._generate_sample_metadata()
    
    def _generate_sample_metadata(self) -> List[MetadataModel]:
        metadata_list = []
        
        categories = {
            "text": ["news", "review", "article", "story", "technical", "blog"],
            "image": ["nature", "technology", "art", "food", "people", "architecture"],
            "audio": ["music", "speech", "ambient", "effects", "podcast", "interview"]
        }
        
        sample_counts = {
            "text": 30,
            "image": 40, 
            "audio": 20
        }
        
        for modality, count in sample_counts.items():
            for i in range(count):
                # Generate a random category for this modality
                category = random.choice(categories[modality])
                
                # Generate a sample file path
                file_name = f"{uuid.uuid4()}.{self._get_extension(modality)}"
                file_path = os.path.join(self.data_dir, modality, file_name)
                
                # Create file with sample content if it doesn't exist
                if not os.path.exists(file_path):
                    self._create_sample_file(file_path, modality)
                
                # Create metadata
                metadata = MetadataModel(
                    id=str(uuid.uuid4()),
                    title=f"Sample {category.title()} {modality.title()} {i+1}",
                    description=f"This is a sample {modality} item in the {category} category",
                    category=category,
                    modality=modality,
                    source_path=file_path,
                    tags=[modality, category, "sample"],
                    created_at=self._random_date()
                )
                
                metadata_list.append(metadata)
        
        logger.info(f"Generated {len(metadata_list)} sample metadata items")
        return metadata_list
    
    def _create_sample_file(self, file_path: str, modality: str):
        """Create a sample file based on modality"""
        if modality == "text":
            # Create a simple text file
            with open(file_path, 'w') as f:
                f.write(f"This is sample text content for {os.path.basename(file_path)}.\n")
                f.write("It contains multiple lines of text.\n")
                f.write("This is used for testing the multi-modal search system.\n")
        
        elif modality == "image":
            # Create a tiny image file (1x1 pixel)
            try:
                import numpy as np
                from PIL import Image
                
                # Create a 100x100 pixel image with random color
                img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(file_path)
            except ImportError:
                # If PIL is not available, create an empty file
                with open(file_path, 'wb') as f:
                    f.write(b'\x00' * 100)  # Write some bytes
        
        elif modality == "audio":
            # Create an empty audio file
            with open(file_path, 'wb') as f:
                f.write(b'\x00' * 100)  # Write some bytes
    
    def _get_extension(self, modality: str) -> str:
        """Get file extension based on modality"""
        if modality == "text":
            return "txt"
        elif modality == "image":
            return "jpg"
        elif modality == "audio":
            return "wav"
        else:
            return "bin"
    
    def _random_date(self) -> str:
        """Generate a random date in the past year"""
        import datetime
        days = random.randint(0, 365)
        date = datetime.datetime.now() - datetime.timedelta(days=days)
        return date.isoformat()


class IngestionService:
    """Service for ingesting and processing data for the multi-modal search system"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.embedding_service = EmbeddingService()
        self.storage_service = StorageService(use_mock_if_unavailable=True)
    
    async def process_text_batch(self, metadata_batch: List[MetadataModel]):
        """Process a batch of text items"""
        try:
            # Extract text from files
            texts = []
            for metadata in metadata_batch:
                try:
                    if os.path.exists(metadata.source_path):
                        with open(metadata.source_path, 'r') as f:
                            text = f.read()
                    else:
                        # If file doesn't exist, use title and description
                        text = f"{metadata.title}. {metadata.description}"
                    texts.append(text)
                except Exception as e:
                    logger.error(f"Error reading text file {metadata.source_path}: {e}")
                    # Use metadata as fallback
                    texts.append(f"{metadata.title}. {metadata.description}")
            
            # Generate embeddings
            result = await self.embedding_service.process_text(texts)
            embeddings = result["embeddings"]
            
            # Store metadata and embeddings
            metadata_ids = []
            item_ids = []
            
            for i, metadata in enumerate(metadata_batch):
                try:
                    # Store metadata
                    metadata_id = self.storage_service.store_metadata(metadata)
                    metadata_ids.append(metadata_id)
                    item_ids.append(metadata.id)
                except Exception as e:
                    logger.error(f"Error storing metadata for text item: {e}")
            
            # Batch store embeddings if we have any
            if item_ids and metadata_ids:
                try:
                    self.storage_service.batch_store_embeddings(
                        item_ids, metadata_ids, embeddings, "text"
                    )
                except Exception as e:
                    logger.error(f"Error storing text embeddings: {e}")
            
            return {
                "processed": len(metadata_batch),
                "metrics": result.get("metrics", {})
            }
        except Exception as e:
            logger.error(f"Error in process_text_batch: {e}")
            return {"processed": 0, "error": str(e)}
    
    async def process_image_batch(self, metadata_batch: List[MetadataModel]):
        """Process a batch of image items"""
        try:
            # Collect valid image paths
            valid_metadata = []
            image_paths = []
            
            for metadata in metadata_batch:
                if os.path.exists(metadata.source_path):
                    valid_metadata.append(metadata)
                    image_paths.append(metadata.source_path)
                else:
                    logger.warning(f"Image file not found: {metadata.source_path}")
            
            if not image_paths:
                logger.warning("No valid images found in batch")
                return {"processed": 0}
            
            # Generate embeddings
            result = await self.embedding_service.process_images(image_paths)
            embeddings = result["embeddings"]
            
            # Store metadata and embeddings
            metadata_ids = []
            item_ids = []
            
            for i, metadata in enumerate(valid_metadata):
                try:
                    # Store metadata
                    metadata_id = self.storage_service.store_metadata(metadata)
                    metadata_ids.append(metadata_id)
                    item_ids.append(metadata.id)
                except Exception as e:
                    logger.error(f"Error storing metadata for image item: {e}")
            
            # Batch store embeddings if we have any
            if item_ids and metadata_ids:
                try:
                    self.storage_service.batch_store_embeddings(
                        item_ids, metadata_ids, embeddings, "image"
                    )
                except Exception as e:
                    logger.error(f"Error storing image embeddings: {e}")
            
            return {
                "processed": len(valid_metadata),
                "metrics": result.get("metrics", {})
            }
        except Exception as e:
            logger.error(f"Error in process_image_batch: {e}")
            return {"processed": 0, "error": str(e)}
    
    async def process_audio_batch(self, metadata_batch: List[MetadataModel]):
        """Process a batch of audio items"""
        try:
            # Collect valid audio paths
            valid_metadata = []
            audio_paths = []
            
            for metadata in metadata_batch:
                if os.path.exists(metadata.source_path):
                    valid_metadata.append(metadata)
                    audio_paths.append(metadata.source_path)
                else:
                    logger.warning(f"Audio file not found: {metadata.source_path}")
            
            if not audio_paths:
                logger.warning("No valid audio files found in batch")
                return {"processed": 0}
            
            # Generate embeddings
            result = await self.embedding_service.process_audio(audio_paths)
            embeddings = result["embeddings"]
            
            # Store metadata and embeddings
            metadata_ids = []
            item_ids = []
            
            for i, metadata in enumerate(valid_metadata):
                try:
                    # Store metadata
                    metadata_id = self.storage_service.store_metadata(metadata)
                    metadata_ids.append(metadata_id)
                    item_ids.append(metadata.id)
                except Exception as e:
                    logger.error(f"Error storing metadata for audio item: {e}")
            
            # Batch store embeddings if we have any
            if item_ids and metadata_ids:
                try:
                    self.storage_service.batch_store_embeddings(
                        item_ids, metadata_ids, embeddings, "audio"
                    )
                except Exception as e:
                    logger.error(f"Error storing audio embeddings: {e}")
            
            return {
                "processed": len(valid_metadata),
                "metrics": result.get("metrics", {})
            }
        except Exception as e:
            logger.error(f"Error in process_audio_batch: {e}")
            return {"processed": 0, "error": str(e)}
    
    async def batch_ingest(self, batch_size=16, sample_mode=True):
        """Ingest data in batches"""
        try:
            # Check if MongoDB is available
            if not hasattr(self.storage_service, 'db') or self.storage_service.db is None:
                logger.error("MongoDB not available")
                return {"status": "error", "message": "MongoDB not available"}
            
            start_time = time.time()
            
            # Load all metadata
            all_metadata = self.data_loader.download_all_data(sample_mode=sample_mode)
            
            # Group by modality
            modality_groups = {}
            for metadata in all_metadata:
                if metadata.modality not in modality_groups:
                    modality_groups[metadata.modality] = []
                modality_groups[metadata.modality].append(metadata)
            
            # Process each modality
            results = {}
            
            # Process text
            if "text" in modality_groups:
                logger.info(f"Processing {len(modality_groups['text'])} text items")
                text_results = []
                for i in range(0, len(modality_groups["text"]), batch_size):
                    batch = modality_groups["text"][i:i+batch_size]
                    result = await self.process_text_batch(batch)
                    text_results.append(result)
                results["text"] = text_results
            
            # Process images
            if "image" in modality_groups:
                logger.info(f"Processing {len(modality_groups['image'])} image items")
                image_results = []
                for i in range(0, len(modality_groups["image"]), batch_size):
                    batch = modality_groups["image"][i:i+batch_size]
                    result = await self.process_image_batch(batch)
                    image_results.append(result)
                results["image"] = image_results
            
            # Process audio
            if "audio" in modality_groups:
                logger.info(f"Processing {len(modality_groups['audio'])} audio items")
                audio_results = []
                for i in range(0, len(modality_groups["audio"]), batch_size):
                    batch = modality_groups["audio"][i:i+batch_size]
                    result = await self.process_audio_batch(batch)
                    audio_results.append(result)
                results["audio"] = audio_results
            
            total_time = time.time() - start_time
            
            # Calculate total processed items
            total_processed = sum(
                sum(result.get("processed", 0) for result in results.get(modality, []))
                for modality in results
            )
            
            logger.info(f"Batch ingestion completed: {total_processed} items in {total_time:.2f} seconds")
            
            return {
                "status": "success",
                "total_processed": total_processed,
                "total_time": total_time,
                "average_throughput": total_processed / total_time if total_time > 0 else 0,
                "results_by_modality": results
            }
            
        except Exception as e:
            logger.error(f"Error in batch_ingest: {e}")
            return {"status": "error", "message": f"Error during ingestion: {str(e)}"}
    
    async def generate_sample_data(self, count=20):
        """Generate synthetic sample data for demonstration"""
        try:
            # Check if MongoDB is available
            if not hasattr(self.storage_service, 'db') or self.storage_service.db is None:
                logger.error("MongoDB not available")
                return {"status": "error", "message": "MongoDB not available"}
            
            # Sample data categories and modalities
            categories = {
                "text": ["news", "review", "article", "blog"],
                "image": ["nature", "technology", "art", "food"],
                "audio": ["music", "speech", "ambient", "podcast"]
            }
            
            # Sample titles and descriptions by category
            sample_content = {
                "news": ["Breaking News", "Latest Updates", "World Events", "Local News"],
                "review": ["Product Review", "Movie Review", "Book Review", "Game Review"],
                "article": ["Scientific Article", "Technical Article", "Opinion Piece", "Feature Article"],
                "blog": ["Personal Blog", "Tech Blog", "Travel Blog", "Food Blog"],
                "nature": ["Mountains", "Beaches", "Forests", "Wildlife"],
                "technology": ["Gadgets", "Software", "Robots", "AI"],
                "art": ["Paintings", "Sculptures", "Digital Art", "Photography"],
                "food": ["Cuisine", "Recipes", "Restaurants", "Ingredients"],
                "music": ["Classical", "Rock", "Jazz", "Electronic"],
                "speech": ["Lecture", "Interview", "Podcast", "Presentation"],
                "ambient": ["Nature Sounds", "City Ambience", "White Noise", "ASMR"],
                "podcast": ["Tech Podcast", "News Podcast", "Comedy Podcast", "Educational Podcast"]
            }
            
            # Generate sample embeddings
            all_embeddings = np.random.randn(count, 1024).astype(np.float32)
            
            # Generate items across modalities
            items_per_modality = count // len(categories)
            leftover = count % len(categories)
            
            modality_counts = {
                "text": items_per_modality + (1 if leftover > 0 else 0),
                "image": items_per_modality + (1 if leftover > 1 else 0),
                "audio": items_per_modality + (1 if leftover > 2 else 0),
            }
            
            total_created = 0
            
            # Create items for each modality
            for modality, modality_count in modality_counts.items():
                modality_cats = categories[modality]
                
                for i in range(modality_count):
                    # Select a category for this item
                    category = modality_cats[i % len(modality_cats)]
                    
                    # Select content based on category
                    content_options = sample_content[category]
                    content_type = content_options[i % len(content_options)]
                    
                    # Create metadata
                    metadata = MetadataModel(
                        id=str(uuid.uuid4()),
                        title=f"{content_type} {i+1}",
                        description=f"Sample {category} content in {modality} format",
                        category=category,
                        modality=modality,
                        source_path=f"data/raw/{modality}/{uuid.uuid4()}.{self._get_extension(modality)}",
                        tags=[modality, category, content_type.lower().replace(" ", "_")],
                        created_at=self._random_date()
                    )
                    
                    # Store metadata
                    try:
                        metadata_id = self.storage_service.store_metadata(metadata)
                        
                        # Store embedding
                        self.storage_service.store_embedding(
                            item_id=metadata.id,
                            metadata_id=metadata_id,
                            embedding=all_embeddings[total_created],
                            modality=modality
                        )
                        
                        total_created += 1
                    except Exception as e:
                        logger.error(f"Error storing sample item: {e}")
            
            logger.info(f"Created {total_created} sample items")
            
            return {
                "status": "success",
                "created": total_created,
                "by_modality": modality_counts
            }
            
        except Exception as e:
            logger.error(f"Error generating sample data: {e}")
            return {"status": "error", "message": str(e)}
    
    def _get_extension(self, modality: str) -> str:
        """Get file extension for modality"""
        if modality == "text":
            return "txt"
        elif modality == "image":
            return "jpg"
        elif modality == "audio":
            return "wav"
        else:
            return "bin"
    
    def _random_date(self) -> str:
        """Generate a random date within the past year"""
        import datetime
        import random
        
        days = random.randint(0, 365)
        date = datetime.datetime.now() - datetime.timedelta(days=days)
        return date.isoformat()