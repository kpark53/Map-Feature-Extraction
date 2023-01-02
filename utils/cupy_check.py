import cupy

dev = cupy.cuda.Device()
print('Compute Capability', dev.compute_capability)
print('GPU Memory', dev.mem_info)

import torch
print(torch.cuda.is_available())
