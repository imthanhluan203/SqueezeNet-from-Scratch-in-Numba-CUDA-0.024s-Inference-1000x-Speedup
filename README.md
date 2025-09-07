# SqueezeNet from Scratch with Numba CUDA

Welcome to an optimized, high-performance implementation of SqueezeNet, meticulously crafted from the ground up using Python and Numba CUDA. This project delivers an impressive 92% accuracy on the Tomato Disease dataset, achieving a blazing-fast 0.02s/image inference time and a 1000x speedup compared to NumPy. Harnessing the power of CUDA, this implementation is designed for speed, scalability, and ease of use.

## Project Overview

This project reimagines SqueezeNet with a focus on performance and modularity. By leveraging CUDA-accelerated operations, it achieves unparalleled efficiency in training and inference. The codebase is structured for clarity, maintainability, and extensibility, making it ideal for researchers and developers working on deep learning tasks with GPU acceleration.

## Results

- **Accuracy**: Achieves 92% accuracy on the Tomato Disease dataset.
- **Inference Speed**: Processes images in just 0.024 seconds, a 1000x speedup over NumPy.
- **Optimization**: Fully CUDA-accelerated operations for maximum performance on NVIDIA GPUs.

## Directory Structure

```
src/
├── config.py              # Configuration settings and constants
├── utils.py               # Utility functions and shared imports
├── data.py                # Data loading and preprocessing utilities
├── model.py               # SqueezeNet model architecture
├── backward.py            # Backward pass implementations
├── train.py               # Training logic and workflow
├── main.py                # Main entry point for execution
├── cuda_ops/              # CUDA-accelerated operations
    ├── __init__.py
    ├── softmax.py         # Softmax layer operations
    ├── pool.py            # Pooling layer operations
    ├── conv.py            # Convolution layer operations
    ├── relu.py            # ReLU activation operations
    ├── fire_module.py     # Fire module operations
    └── utils.py           # CUDA utility functions
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the CUDA toolkit installed and an NVIDIA GPU with CUDA support.

## Usage

### Running Training
Kickstart training with a single command:
```python
from src.main import main

# Launch training
weights, loss_history, acc_history = main()
```

### Using Individual Modules
For fine-grained control, you can work with specific modules:
```python
# Load and preprocess data
from src.data import load_datasets, create_data_loaders
train_ds, test_ds, categories = load_datasets()
train_loader, val_loader = create_data_loaders(train_ds, test_ds)

# Train the model
from src.train import train_model
weights, loss_history, acc_history = train_model(train_loader)

# Directly use CUDA operations
from src.cuda_ops.conv import run_conv2d_cuda
from src.cuda_ops.relu import run_Relu_Cuda
# ... additional CUDA operations
```

## Configuration

Customize your training setup by editing `src/config.py` to tweak:
- Dataset paths
- Batch size
- Learning rate
- Number of epochs

## Key Features

1. **CUDA Acceleration**: All core operations are optimized for NVIDIA GPUs using CUDA, delivering lightning-fast performance.
2. **Modular Architecture**: Cleanly separated modules ensure easy maintenance and extensibility.
3. **Efficient Memory Management**: Automatic GPU memory handling for seamless operation.

## CUDA Operations

### Convolution
- `run_conv2d_cuda()`: High-performance 2D convolution
- `run_conv2d_cuda_shared_2()`: Optimized 2D convolution with shared memory
- `run_conv_backward()`: Backward pass for convolution layers

### Pooling
- `run_maxpool2d_cuda()`: Fast 2D max pooling
- `run_global_average_Cuda()`: Global average pooling
- `run_maxpooling_backward()`: Backward pass for max pooling

### Activation
- `run_Relu_Cuda()`: Efficient ReLU activation
- `run_relu_backward()`: Backward pass for ReLU

### Softmax
- `run_softmax_cuda()`: Optimized softmax operation
- `run_softmax_backward()`: Backward pass for softmax

### Fire Module
- `fire_module_cuda()`: Forward pass for SqueezeNet's Fire module
- `run_firemodule_backward()`: Backward pass for Fire module

## Notes

- Ensure your GPU has sufficient memory to handle the selected batch size.
- The code is fine-tuned for NVIDIA GPUs with CUDA support.
- Adjust TPB (Threads Per Block) in CUDA operations based on your GPU's specifications for optimal performance.
