import torch
import numpy as np

TORCH_DTYPE = torch.float32
NUMPY_DTYPE = np.float32

WARMUP = 1
TRIALS = 10

torch.set_num_interop_threads(1)
# torch.set_num_threads(1)
