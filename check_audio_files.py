# check_audio_files.py
import os
from app.storage.service import StorageService

# Check physical audio files
audio_dir = "data/raw/audio"
if os.path.exists(audio_dir):
    audio_files = os.listdir(audio_dir)
    print(f"Found {len(audio_files)} audio files in {audio_dir}")
    for i, file in enumerate(audio_files[:5]):
        print(f"  {i+1}: {file}")
else:
    print(f"Audio directory {audio_dir} doesn't exist")

# Check audio entries in database
storage = StorageService()
audio_docs = list(storage.metadata_collection.find({"modality": "audio"}).limit(5))
print(f"\nFound {storage.metadata_collection.count_documents({'modality': 'audio'})} audio documents in database")
for i, doc in enumerate(audio_docs):
    print(f"  {i+1}: {doc.get('title')} - {doc.get('source_path')}")