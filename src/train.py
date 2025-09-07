"""
Training script for SqueezeNet CUDA implementation
"""
import math
import numpy as np
from numba import cuda
from tqdm import tqdm
from .config import EPOCHS, MODEL_LOAD_PATH, MODEL_SAVE_PATH
from .utils import get_memory_info, reset_cuda_device
from .model import squeezenet_forward_cuda
from .backward import backward_squeezenet_cuda
from .cuda_ops.softmax import run_softmax_cuda

def load_weights():
    """Load pre-trained weights"""
    load_state_5 = np.load(MODEL_LOAD_PATH, allow_pickle=True)
    weights_np = {k: v for k, v in load_state_5["weights"].item().items()}
    return weights_np

def calculate_loss(X_output, Y_cuda):
    """Calculate cross-entropy loss"""
    epsilon = 1e-20
    batch_loss = 0.0
    for m in range(X_output.shape[0]):
        for n in range(X_output.shape[1]):
            if Y_cuda[m][n] == 1:
                batch_loss -= math.log(X_output[m][n] + epsilon)
    return batch_loss

def calculate_accuracy(X_output, y_np):
    """Calculate accuracy"""
    output_cpu = X_output.copy_to_host()
    predicted = np.argmax(output_cpu, axis=1)
    return np.sum(predicted == y_np)

def train_epoch(epoch, epochs, train_loader, weights_np):
    """Train for one epoch"""
    print(weights_np["classifier.0.bias"].flatten())
    print(get_memory_info())
    print(f"\nEpoch {epoch + 1}/{epochs}")
    
    weights_cuda = {k: cuda.to_device(v) for k, v in weights_np.items()}
    running_loss, running_corr, running_samples = 0.0, 0, 0
    
    train_iter = tqdm(train_loader, desc=f"[Epoch {epoch}/{epochs}] Training", leave=False)
    
    for imgs, labels in train_iter:
        from .data import prepare_batch_data
        x_cuda, Y_cuda, y_np = prepare_batch_data(imgs, labels)
        
        # Forward pass
        x_cuda_forward = squeezenet_forward_cuda(x_cuda, weights_cuda)
        X_output = run_softmax_cuda(x_cuda_forward["avgpool10"])

        # Calculate loss
        batch_loss = calculate_loss(X_output, Y_cuda)
        running_loss += batch_loss
       
        # Calculate accuracy
        running_corr += calculate_accuracy(X_output, y_np)
        running_samples += y_np.shape[0]
        
        memory = get_memory_info()
        train_iter.set_postfix(loss=running_loss / running_samples,
                               acc=running_corr / running_samples,
                               freemem=memory)
        
        # Backward pass
        weights_cuda = backward_squeezenet_cuda(x_cuda, x_cuda_forward, weights_cuda, Y_cuda)
    
    train_loss = running_loss / running_samples
    train_acc = running_corr / running_samples
    
    # Convert weights back to CPU
    weights_np = {k: v.copy_to_host() for k, v in weights_cuda.items()}
    reset_cuda_device()
    
    return train_loss, train_acc, weights_np

def train_model(train_loader):
    """Main training function"""
    weights_np = load_weights()
    Loss_container = []
    Acc_container = []
    
    for epoch in range(EPOCHS):
        train_loss, train_acc, weights_np = train_epoch(epoch, EPOCHS, train_loader, weights_np)
        
        Loss_container.append(train_loss)
        Acc_container.append(train_acc)
        
        print(f"Epoch {epoch + 1}/{EPOCHS} | "
              f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(Loss_container)
        print(Acc_container)
    
    # Save final model
    np.savez(MODEL_SAVE_PATH, weights=weights_np)
    
    return weights_np, Loss_container, Acc_container
