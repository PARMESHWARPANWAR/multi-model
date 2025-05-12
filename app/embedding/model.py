import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

class EmbeddingModel:
    def __init__(self):
        # Load the model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize ImageBind model
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval()
        self.model.to(self.device)
        
        # Set batch sizes based on GPU memory
        self.batch_sizes = {
            ModalityType.TEXT: 256,
            ModalityType.VISION: 64,
            ModalityType.AUDIO: 32,
            # Add other modalities as needed
        }
        
    def generate_embedding(self, modality_type, inputs, batch_size=None):
        """
        Generate embeddings for inputs of a specific modality
        """
        # Use default batch size if not specified
        if batch_size is None:
            batch_size = self.batch_sizes.get(modality_type, 16)
        
        # Process in batches
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(inputs), batch_size):
                batch = inputs[i:i+batch_size]
                inputs_dict = {modality_type: batch}
                
                # Generate embeddings for batch
                embeddings = self.model(inputs_dict)
                embeddings = embeddings[modality_type].cpu().numpy()
                all_embeddings.append(embeddings)
        
        return torch.cat(all_embeddings, dim=0) if len(all_embeddings) > 1 else all_embeddings[0]