# reingest_data.py
import asyncio
from app.ingestion.service import IngestionService

async def reingest():
    service = IngestionService()
    
    # This will create and ingest sample data including audio
    result = await service.batch_ingest(batch_size=16, sample_mode=True)
    print(f"Ingestion result: {result}")
    
    # Or generate more sample data
    sample_result = await service.generate_sample_data(count=30)
    print(f"Sample generation result: {sample_result}")

if __name__ == "__main__":
    asyncio.run(reingest())