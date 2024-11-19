import torch

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

# Verify PyTorch is using CUDA 11.6
print("CUDA version used by PyTorch:", torch.version.cuda)

# Get the name of the detected GPU (if available)
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
