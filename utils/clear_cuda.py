'''Purpose: Clears the CUDA cache to free up GPU memory.
Key Functionality: Executes torch.cuda.empty_cache().'''
import torch
torch.cuda.empty_cache()
print("✅ CUDA cache cleared")
