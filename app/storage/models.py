from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class MetadataModel(BaseModel):
    """Model for item metadata"""
    id: str
    title: str
    description: Optional[str] = None
    tags: List[str] = []
    category: Optional[str] = None
    modality: str
    source_path: str
    created_at: datetime = Field(default_factory=datetime.now)
    
class EmbeddingDocument(BaseModel):
    """Model for storing embedding information"""
    id: str
    metadata_id: str
    modality: str
    dimension: int
    # The actual embedding vector is stored in FAISS
    created_at: datetime = Field(default_factory=datetime.now)