import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def gpu_info():
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        used = torch.cuda.memory_allocated(0) / 1e9
        total = props.total_memory / 1e9
        print(f"GPU: {props.name}  {used:.1f}/{total:.1f} GB used")
    else:
        print("No GPU detected — training will be slow.")
