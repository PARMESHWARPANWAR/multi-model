import os
import json
import uuid
import logging
import requests
import zipfile
import tarfile
import time
import asyncio
from typing import List, Dict, Any
import numpy as np
from PIL import Image
import soundfile as sf
import wave
from tqdm import tqdm
import pandas as pd
from app.storage.models import MetadataModel
from app.embedding.service import EmbeddingService
from app.storage.service import StorageService

logger = logging.getLogger(__name__)

class RealDataLoader:
    """Data loader for downloading real datasets for multi-modal search"""
    
    def __init__(self, base_data_dir="data/raw"):
        self.base_data_dir = base_data_dir
        os.makedirs(base_data_dir, exist_ok=True)
        
        # Create subdirectories for each modality
        self.modality_dirs = {
            "text": os.path.join(base_data_dir, "text"),
            "image": os.path.join(base_data_dir, "image"),
            "audio": os.path.join(base_data_dir, "audio"),
            "video": os.path.join(base_data_dir, "video")
        }
        
        for dir_path in self.modality_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def download_imagenet_subset(self, num_samples=5000):
        """Download a subset of ImageNet images using ImageNet-Mini or similar sources"""
        print(f"Downloading {num_samples} images from ImageNet subset")
        
        image_dir = self.modality_dirs["image"]
        metadata = []
        
        # Use ImageNet-Mini or other publicly available subset
        # Alternative: Use CIFAR-100 or other image datasets
        try:
            # Download CIFAR-100 as an alternative
            cifar_url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
            cifar_path = os.path.join(self.base_data_dir, "cifar-100.tar.gz")
            
            if not os.path.exists(cifar_path):
                print("Downloading CIFAR-100 dataset...")
                response = requests.get(cifar_url, stream=True)
                with open(cifar_path, 'wb') as f:
                    for chunk in tqdm(response.iter_content(chunk_size=8192)):
                        f.write(chunk)
                
                # Extract the dataset
                with tarfile.open(cifar_path, 'r:gz') as tar:
                    tar.extractall(self.base_data_dir)
            
            # Load CIFAR-100 data
            import pickle
            with open(os.path.join(self.base_data_dir, "cifar-100-python", "train"), 'rb') as f:
                train_data = pickle.load(f, encoding='bytes')
            
            images = train_data[b'data']
            labels = train_data[b'fine_labels']
            
            # Process a subset of images
            for i in range(min(num_samples, len(images))):
                # Reshape CIFAR image
                img_array = images[i].reshape(3, 32, 32).transpose(1, 2, 0)
                img = Image.fromarray(img_array, 'RGB')
                
                # Resize to 224x224 for consistency with ImageNet
                img = img.resize((224, 224), Image.BICUBIC)
                
                # Save image
                image_id = f"cifar_{i:05d}"
                image_path = os.path.join(image_dir, f"{image_id}.jpg")
                img.save(image_path)
                
                # Create metadata
                item_metadata = MetadataModel(
                    id=str(uuid.uuid4()),
                    title=f"CIFAR Image {i}",
                    description=f"Image from CIFAR-100 dataset, label: {labels[i]}",
                    tags=["cifar", "image", f"class_{labels[i]}"],
                    category=f"class_{labels[i]}",
                    modality="image",
                    source_path=image_path,
                    created_at=time.strftime("%Y-%m-%dT%H:%M:%S")
                )
                
                metadata.append(item_metadata)
                
        except Exception as e:
            logger.error(f"Error downloading CIFAR-100: {e}")
            print("Falling back to synthetic images...")
            return self._generate_synthetic_images(num_samples)
        
        return metadata
    
    def download_audioset_subset(self, num_samples=2000):
        """Download a subset of audio clips from FreeSound or similar sources"""
        print(f"Downloading {num_samples} audio clips")
        
        audio_dir = self.modality_dirs["audio"]
        metadata = []
        
        # Use FreeSound API or ESC-50 dataset as alternative
        try:
            # Download ESC-50 dataset (Environmental Sound Classification)
            esc50_url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
            esc50_path = os.path.join(self.base_data_dir, "esc50.zip")
            
            if not os.path.exists(esc50_path):
                print("Downloading ESC-50 dataset...")
                response = requests.get(esc50_url, stream=True)
                with open(esc50_path, 'wb') as f:
                    for chunk in tqdm(response.iter_content(chunk_size=8192)):
                        f.write(chunk)
                
                # Extract the dataset
                with zipfile.ZipFile(esc50_path, 'r') as zip_ref:
                    zip_ref.extractall(self.base_data_dir)
            
            # Load ESC-50 metadata
            metadata_path = os.path.join(self.base_data_dir, "ESC-50-master", "meta", "esc50.csv")
            esc_metadata = pd.read_csv(metadata_path)
            
            # Process audio files
            audio_source_dir = os.path.join(self.base_data_dir, "ESC-50-master", "audio")
            
            for i, row in esc_metadata.iterrows():
                if i >= num_samples:
                    break
                
                source_file = os.path.join(audio_source_dir, row['filename'])
                if os.path.exists(source_file):
                    # Copy to our audio directory
                    audio_id = f"esc50_{i:05d}"
                    target_path = os.path.join(audio_dir, f"{audio_id}.wav")
                    
                    # Read and resave audio to ensure consistent format
                    data, samplerate = sf.read(source_file)
                    sf.write(target_path, data, samplerate)
                    
                    # Create metadata
                    item_metadata = MetadataModel(
                        id=str(uuid.uuid4()),
                        title=f"ESC-50 Audio {row['target']}",
                        description=f"Environmental sound: {row['category']}",
                        tags=["esc50", "audio", row['category'], row['target']],
                        category=row['category'],
                        modality="audio",
                        source_path=target_path,
                        created_at=time.strftime("%Y-%m-%dT%H:%M:%S")
                    )
                    
                    metadata.append(item_metadata)
                    
        except Exception as e:
            logger.error(f"Error downloading ESC-50: {e}")
            print("Falling back to synthetic audio...")
            return self._generate_synthetic_audio(num_samples)
        
        return metadata
    
    def download_imdb_reviews(self, num_samples=3000):
        """Download IMDB movie reviews dataset"""
        print(f"Downloading {num_samples} text samples from IMDB")
        
        text_dir = self.modality_dirs["text"]
        metadata = []
        
        try:
            # Download IMDB dataset
            imdb_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
            imdb_path = os.path.join(self.base_data_dir, "aclImdb_v1.tar.gz")
            
            if not os.path.exists(imdb_path):
                print("Downloading IMDB dataset...")
                response = requests.get(imdb_url, stream=True)
                with open(imdb_path, 'wb') as f:
                    for chunk in tqdm(response.iter_content(chunk_size=8192)):
                        f.write(chunk)
                
                # Extract the dataset
                with tarfile.open(imdb_path, 'r:gz') as tar:
                    tar.extractall(self.base_data_dir)
            
            # Process reviews
            imdb_dir = os.path.join(self.base_data_dir, "aclImdb")
            count = 0
            
            for split in ['train', 'test']:
                for sentiment in ['pos', 'neg']:
                    review_dir = os.path.join(imdb_dir, split, sentiment)
                    
                    if not os.path.exists(review_dir):
                        continue
                    
                    for review_file in os.listdir(review_dir):
                        if count >= num_samples:
                            break
                        
                        if review_file.endswith('.txt'):
                            source_path = os.path.join(review_dir, review_file)
                            
                            # Read review
                            with open(source_path, 'r', encoding='utf-8') as f:
                                review_text = f.read()
                            
                            # Save to our text directory
                            text_id = f"imdb_{count:05d}"
                            target_path = os.path.join(text_dir, f"{text_id}.txt")
                            
                            with open(target_path, 'w', encoding='utf-8') as f:
                                f.write(review_text)
                            
                            # Create metadata
                            item_metadata = MetadataModel(
                                id=str(uuid.uuid4()),
                                title=f"IMDB Review {count}",
                                description=f"Movie review with {sentiment} sentiment",
                                tags=["imdb", "text", "review", sentiment],
                                category="review",
                                modality="text",
                                source_path=target_path,
                                created_at=time.strftime("%Y-%m-%dT%H:%M:%S")
                            )
                            
                            metadata.append(item_metadata)
                            count += 1
                    
                    if count >= num_samples:
                        break
                if count >= num_samples:
                    break
                    
        except Exception as e:
            logger.error(f"Error downloading IMDB: {e}")
            print("Falling back to synthetic text...")
            return self._generate_synthetic_text(num_samples)
        
        return metadata
    
    def download_youtube8m_subset(self, num_samples=500):
        """Download a subset of YouTube-8M video features (not actual videos)"""
        print(f"Note: YouTube-8M provides features, not raw videos")
        
        video_dir = self.modality_dirs["video"]
        metadata = []
        
        # For demo purposes, we'll create placeholder video metadata
        # In production, you'd download actual video features from YouTube-8M
        
        # Create synthetic video metadata
        categories = ["sports", "music", "news", "education", "entertainment"]
        
        for i in range(num_samples):
            video_id = f"video_{i:05d}"
            video_path = os.path.join(video_dir, f"{video_id}.json")
            
            # Create placeholder video feature file
            video_features = {
                "id": video_id,
                "duration": np.random.randint(30, 300),
                "features": np.random.randn(1024).tolist()
            }
            
            with open(video_path, 'w') as f:
                json.dump(video_features, f)
            
            # Create metadata
            category = np.random.choice(categories)
            item_metadata = MetadataModel(
                id=str(uuid.uuid4()),
                title=f"Video {i}",
                description=f"Sample video from {category} category",
                tags=["video", category, "youtube8m"],
                category=category,
                modality="video",
                source_path=video_path,
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S")
            )
            
            metadata.append(item_metadata)
        
        return metadata
    
    def _generate_synthetic_images(self, num_samples):
        """Fallback to generate synthetic images"""
        image_dir = self.modality_dirs["image"]
        metadata = []
        
        for i in range(num_samples):
            # Create a colored image
            img = Image.new('RGB', (224, 224), 
                          color=(np.random.randint(0, 255), 
                                np.random.randint(0, 255), 
                                np.random.randint(0, 255)))
            
            image_id = f"synthetic_{i:05d}"
            image_path = os.path.join(image_dir, f"{image_id}.jpg")
            img.save(image_path)
            
            item_metadata = MetadataModel(
                id=str(uuid.uuid4()),
                title=f"Synthetic Image {i}",
                description=f"Synthetic image for testing",
                tags=["synthetic", "image", "test"],
                category="synthetic",
                modality="image",
                source_path=image_path,
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S")
            )
            
            metadata.append(item_metadata)
        
        return metadata
    
    def _generate_synthetic_audio(self, num_samples):
        """Fallback to generate synthetic audio"""
        audio_dir = self.modality_dirs["audio"]
        metadata = []
        
        for i in range(num_samples):
            # Generate a simple sine wave
            duration = 1.0  # seconds
            sample_rate = 44100
            frequency = 440 * (i % 12 + 1)  # Different frequencies
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            waveform = 0.5 * np.sin(2 * np.pi * frequency * t)
            
            audio_id = f"synthetic_{i:05d}"
            audio_path = os.path.join(audio_dir, f"{audio_id}.wav")
            
            # Save as WAV file
            sf.write(audio_path, waveform, sample_rate)
            
            item_metadata = MetadataModel(
                id=str(uuid.uuid4()),
                title=f"Synthetic Audio {i}",
                description=f"Synthetic audio at {frequency}Hz",
                tags=["synthetic", "audio", "test"],
                category="synthetic",
                modality="audio",
                source_path=audio_path,
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S")
            )
            
            metadata.append(item_metadata)
        
        return metadata
    
    def _generate_synthetic_text(self, num_samples):
        """Fallback to generate synthetic text"""
        text_dir = self.modality_dirs["text"]
        metadata = []
        
        for i in range(num_samples):
            text_id = f"synthetic_{i:05d}"
            text_path = os.path.join(text_dir, f"{text_id}.txt")
            
            # Create synthetic text
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(f"This is synthetic text sample {i}.\n")
                f.write(f"It contains multiple sentences for testing.\n")
                f.write(f"The sentiment is {'positive' if i % 2 == 0 else 'negative'}.\n")
            
            item_metadata = MetadataModel(
                id=str(uuid.uuid4()),
                title=f"Synthetic Text {i}",
                description=f"Synthetic text for testing",
                tags=["synthetic", "text", "test"],
                category="synthetic",
                modality="text",
                source_path=text_path,
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S")
            )
            
            metadata.append(item_metadata)
        
        return metadata
    
    def download_all_data(self, 
                        image_samples=5000, 
                        audio_samples=2000, 
                        text_samples=3000,
                        video_samples=0):  # Set to 0 by default since videos are large
        """Download all datasets"""
        all_metadata = []
        
        # Download images
        print("\n--- Downloading Images ---")
        image_metadata = self.download_imagenet_subset(image_samples)
        all_metadata.extend(image_metadata)
        print(f"Downloaded {len(image_metadata)} images")
        
        # Download audio
        print("\n--- Downloading Audio ---")
        audio_metadata = self.download_audioset_subset(audio_samples)
        all_metadata.extend(audio_metadata)
        print(f"Downloaded {len(audio_metadata)} audio files")
        
        # Download text
        print("\n--- Downloading Text ---")
        text_metadata = self.download_imdb_reviews(text_samples)
        all_metadata.extend(text_metadata)
        print(f"Downloaded {len(text_metadata)} text files")
        
        # Download video features (optional)
        if video_samples > 0:
            print("\n--- Downloading Video Features ---")
            video_metadata = self.download_youtube8m_subset(video_samples)
            all_metadata.extend(video_metadata)
            print(f"Downloaded {len(video_metadata)} video features")
        
        # Save metadata index
        metadata_path = os.path.join(self.base_data_dir, "metadata_index.json")
        with open(metadata_path, "w") as f:
            json.dump([m.dict() for m in all_metadata], f, default=str, indent=2)
        
        print(f"\nTotal downloaded: {len(all_metadata)} items")
        print(f"Metadata saved to: {metadata_path}")
        
        return all_metadata


class RealIngestionService:
    """Service for ingesting real data into the multi-modal search system"""
    
    def __init__(self):
        self.data_loader = RealDataLoader()
        self.embedding_service = EmbeddingService()
        self.storage_service = StorageService(use_mock_if_unavailable=False)
    
    async def ingest_batch(self, batch_size=16):
        """Download and ingest real data"""
        try:
            # Check if MongoDB is available
            if not hasattr(self.storage_service, 'db') or self.storage_service.db is None:
                logger.error("MongoDB not available")
                return {"status": "error", "message": "MongoDB not available"}
            
            print("Starting real data ingestion...")
            start_time = time.time()
            
            # Download all data
            all_metadata = self.data_loader.download_all_data(
                image_samples=100,
                audio_samples=100,
                text_samples=200,
                video_samples=10  # Skip videos due to size
            )
            
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
                print(f"\nProcessing {len(modality_groups['text'])} text items...")
                text_results = []
                for i in tqdm(range(0, len(modality_groups["text"]), batch_size)):
                    batch = modality_groups["text"][i:i+batch_size]
                    result = await self._process_text_batch(batch)
                    text_results.append(result)
                results["text"] = text_results
            
            # Process images
            if "image" in modality_groups:
                print(f"\nProcessing {len(modality_groups['image'])} image items...")
                image_results = []
                for i in tqdm(range(0, len(modality_groups["image"]), batch_size)):
                    batch = modality_groups["image"][i:i+batch_size]
                    result = await self._process_image_batch(batch)
                    image_results.append(result)
                results["image"] = image_results
            
            # Process audio
            if "audio" in modality_groups:
                print(f"\nProcessing {len(modality_groups['audio'])} audio items...")
                audio_results = []
                for i in tqdm(range(0, len(modality_groups["audio"]), batch_size)):
                    batch = modality_groups["audio"][i:i+batch_size]
                    result = await self._process_audio_batch(batch)
                    audio_results.append(result)
                results["audio"] = audio_results
            
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
            return {"status": "error", "message": f"Error during ingestion: {str(e)}"}
    
    async def _process_text_batch(self, metadata_batch: List[MetadataModel]):
        """Process a batch of text items"""
        try:
            texts = []
            for metadata in metadata_batch:
                try:
                    with open(metadata.source_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    texts.append(text)
                except Exception as e:
                    logger.error(f"Error reading text file {metadata.source_path}: {e}")
                    texts.append(f"{metadata.title}. {metadata.description}")
            
            # Generate embeddings
            result = await self.embedding_service.process_text(texts)
            embeddings = result["embeddings"]
            
            # Store metadata and embeddings
            metadata_ids = []
            item_ids = []
            
            for i, metadata in enumerate(metadata_batch):
                try:
                    metadata_id = self.storage_service.store_metadata(metadata)
                    metadata_ids.append(metadata_id)
                    item_ids.append(metadata.id)
                except Exception as e:
                    logger.error(f"Error storing metadata: {e}")
            
            if item_ids and metadata_ids:
                try:
                    self.storage_service.batch_store_embeddings(
                        item_ids, metadata_ids, embeddings, "text"
                    )
                except Exception as e:
                    logger.error(f"Error storing embeddings: {e}")
            
            return {"processed": len(metadata_batch)}
        except Exception as e:
            logger.error(f"Error in process_text_batch: {e}")
            return {"processed": 0, "error": str(e)}
    
    async def _process_image_batch(self, metadata_batch: List[MetadataModel]):
        """Process a batch of image items"""
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
            
            # Store metadata and embeddings
            metadata_ids = []
            item_ids = []
            
            for i, metadata in enumerate(valid_metadata):
                try:
                    metadata_id = self.storage_service.store_metadata(metadata)
                    metadata_ids.append(metadata_id)
                    item_ids.append(metadata.id)
                except Exception as e:
                    logger.error(f"Error storing metadata: {e}")
            
            if item_ids and metadata_ids:
                try:
                    self.storage_service.batch_store_embeddings(
                        item_ids, metadata_ids, embeddings, "image"
                    )
                except Exception as e:
                    logger.error(f"Error storing embeddings: {e}")
            
            return {"processed": len(valid_metadata)}
        except Exception as e:
            logger.error(f"Error in process_image_batch: {e}")
            return {"processed": 0, "error": str(e)}
    
    async def _process_audio_batch(self, metadata_batch: List[MetadataModel]):
        """Process a batch of audio items"""
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
            
            # Store metadata and embeddings
            metadata_ids = []
            item_ids = []
            
            for i, metadata in enumerate(valid_metadata):
                try:
                    metadata_id = self.storage_service.store_metadata(metadata)
                    metadata_ids.append(metadata_id)
                    item_ids.append(metadata.id)
                except Exception as e:
                    logger.error(f"Error storing metadata: {e}")
            
            if item_ids and metadata_ids:
                try:
                    self.storage_service.batch_store_embeddings(
                        item_ids, metadata_ids, embeddings, "audio"
                    )
                except Exception as e:
                    logger.error(f"Error storing embeddings: {e}")
            
            return {"processed": len(valid_metadata)}
        except Exception as e:
            logger.error(f"Error in process_audio_batch: {e}")
            return {"processed": 0, "error": str(e)}
