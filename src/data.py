"""
Data loading and preprocessing utilities
"""
import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .config import DIR_TRAIN, DIR_TEST, BATCH_SIZE, SHUFFLE, INPUT_SIZE

def get_transforms():
    """Get data transforms for training and validation"""
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE),
                         interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),              # → FloatTensor trong [0,1]
        transforms.Lambda(lambda x: x * 2.0 - 1.0),  # → FloatTensor trong [-1,1]
    ])
    return transform

def load_datasets():
    """Load training and test datasets"""
    transform = get_transforms()
    
    train_ds = datasets.ImageFolder(DIR_TRAIN, transform=transform)
    test_ds = datasets.ImageFolder(DIR_TEST, transform=transform)
    categories = train_ds.classes  # danh sách tên lớp
    
    return train_ds, test_ds, categories

def create_data_loaders(train_ds, test_ds):
    """Create data loaders for training and validation"""
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
    )

    val_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,           # thường không shuffle validation
    )
    
    return train_loader, val_loader

def prepare_batch_data(imgs, labels):
    """Prepare batch data for CUDA operations"""
    from numba import cuda
    
    x_cuda = cuda.to_device(imgs.cpu().numpy())
    y_np = labels.cpu().numpy()
    
    # One-hot encode
    Y = np.zeros((y_np.size, 10), dtype=np.float32)
    Y[np.arange(y_np.size), y_np] = 1
    Y_cuda = cuda.to_device(Y)
    
    return x_cuda, Y_cuda, y_np

def print_dataset_info(train_loader, val_loader, categories):
    """Print dataset information"""
    print(f'Number of training samples: {len(train_loader)}')
    print(f'Number of validation samples: {len(val_loader)}')
    print(f'Classes: {categories}')
