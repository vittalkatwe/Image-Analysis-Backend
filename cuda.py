import torch

# Check CUDA availability
print(f"CUDA Available: {torch.cuda.is_available()}")

# List available devices
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")

# Test tensor allocation on GPU
try:
    x = torch.tensor([1.0, 2.0, 3.0]).to("cuda")
    print(f"Tensor successfully allocated on GPU: {x}")
except Exception as e:
    print(f"Failed to allocate tensor on GPU: {e}")
