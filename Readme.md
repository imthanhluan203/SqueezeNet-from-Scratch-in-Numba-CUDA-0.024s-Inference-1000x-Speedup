# SqueezeNet CUDA Implementation

Dự án này là một implementation của SqueezeNet sử dụng CUDA operations để tối ưu hóa hiệu suất training.

## Cấu trúc thư mục

```
src/
├
├── config.py              # Cấu hình và constants
├── utils.py               # Utility functions và imports
├── data.py                # Data loading và preprocessing
├── model.py               # SqueezeNet model implementation
├── backward.py            # Backward pass implementation
├── train.py               # Training logic
├── main.py                # Main execution script
├── cuda_ops/              # CUDA operations
    ├── __init__.py
    ├── softmax.py         # Softmax operations
    ├── pool.py            # Pooling operations
    ├── conv.py            # Convolution operations
    ├── relu.py            # ReLU operations
    ├── fire_module.py     # Fire module operations
    └── utils.py           # Utility CUDA operations


## Cài đặt

1. Cài đặt các dependencies:
```bash
pip install -r requirements.txt
```

2. Đảm bảo có CUDA toolkit và GPU hỗ trợ.

## Sử dụng

### Chạy training:
```python
from src.main import main

# Chạy training
weights, loss_history, acc_history = main()
```

### Sử dụng từng module riêng lẻ:

```python
# Load data
from src.data import load_datasets, create_data_loaders
train_ds, test_ds, categories = load_datasets()
train_loader, val_loader = create_data_loaders(train_ds, test_ds)

# Train model
from src.train import train_model
weights, loss_history, acc_history = train_model(train_loader)

# Sử dụng CUDA operations trực tiếp
from src.cuda_ops.conv import run_conv2d_cuda
from src.cuda_ops.relu import run_Relu_Cuda
# ... các operations khác
```

## Cấu hình

Chỉnh sửa file `src/config.py` để thay đổi:
- Đường dẫn dữ liệu
- Batch size
- Learning rate
- Số epochs

## Tính năng chính

1. **CUDA Operations**: Tất cả các operations chính đều được implement trên CUDA
2. **Modular Design**: Code được chia thành các module nhỏ, dễ maintain
3. **Memory Management**: Tự động quản lý memory GPU

## CUDA Operations

### Convolution
- `run_conv2d_cuda()`: 2D convolution cơ bản
- `run_conv2d_cuda_shared_2()`: 2D convolution với shared memory
- `run_conv_backward()`: Backward pass cho convolution

### Pooling
- `run_maxpool2d_cuda()`: Max pooling 2D
- `run_global_average_Cuda()`: Global average pooling
- `run_maxpooling_backward()`: Backward pass cho max pooling

### Activation
- `run_Relu_Cuda()`: ReLU activation
- `run_relu_backward()`: Backward pass cho ReLU

### Softmax
- `run_softmax_cuda()`: Softmax operation
- `run_softmax_backward()`: Backward pass cho softmax

### Fire Module
- `fire_module_cuda()`: Fire module forward pass
- `run_firemodule_backward()`: Fire module backward pass

## Lưu ý

- Đảm bảo GPU có đủ memory cho batch size đã chọn
- Code được tối ưu cho NVIDIA GPUs với CUDA support
- Có thể cần điều chỉnh TPB (Threads Per Block) tùy theo GPU

