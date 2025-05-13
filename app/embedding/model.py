import torch
import numpy as np
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

class EmbeddingModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
       
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval()
        self.model.to(self.device)
        
        self.use_amp = torch.cuda.is_available()
        
        self.batch_sizes = {
            ModalityType.TEXT: 256,
            ModalityType.VISION: 64,
            ModalityType.AUDIO: 32,
        }
    
    def generate_embedding(self, modality_type, inputs, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_sizes.get(modality_type, 16)
      
        if torch.is_tensor(inputs):
            with torch.no_grad():
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        embeddings = self.model({modality_type: inputs})
                else:
                    embeddings = self.model({modality_type: inputs})
                return embeddings[modality_type].cpu().numpy()
       
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(inputs), batch_size):
                batch = inputs[i:i+batch_size]
                inputs_dict = {modality_type: batch}
               
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        embeddings = self.model(inputs_dict)
                else:
                    embeddings = self.model(inputs_dict)
                
                embeddings = embeddings[modality_type].cpu().numpy()
                all_embeddings.append(embeddings)
    
        return np.concatenate(all_embeddings, axis=0) if len(all_embeddings) > 1 else all_embeddings[0]

