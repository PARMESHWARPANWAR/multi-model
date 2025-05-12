# check_modalities.py
from app.storage.service import StorageService

storage = StorageService()

# Count documents by modality
modalities = {}
for doc in storage.metadata_collection.find():
    modality = doc.get("modality", "unknown")
    modalities[modality] = modalities.get(modality, 0) + 1

print("Documents by modality:")
for modality, count in modalities.items():
    print(f"  {modality}: {count}")

# Get distinct modalities
distinct_modalities = storage.metadata_collection.distinct("modality")
print(f"\nDistinct modalities: {distinct_modalities}")