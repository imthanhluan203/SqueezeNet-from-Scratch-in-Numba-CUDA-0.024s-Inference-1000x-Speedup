"""
Utility functions and imports for SqueezeNet CUDA implementation
"""
import os
import random
import numpy as np
import torch
from torchvision import datasets, transforms
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from numba import jit
import numpy as np
from numba import cuda, float32
import time
import torch.nn as nn
import math
from tqdm import tqdm

def setup_seed(seed=42):
    """Setup random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def print_versions():
    """Print versions of key libraries"""
    print('torch', torch.__version__)
    print('numpy', np.__version__)
    print('matplotlib', plt.matplotlib.__version__)

def get_memory_info():
    """Get CUDA memory information"""
    return cuda.current_context().get_memory_info()

def reset_cuda_device():
    """Reset CUDA device to free memory"""
    cuda.get_current_device().reset()
