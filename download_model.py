import os
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

print("Pre-downloading ImageBind model...")
model = imagebind_model.imagebind_huge(pretrained=True)
print("Model downloaded successfully!")