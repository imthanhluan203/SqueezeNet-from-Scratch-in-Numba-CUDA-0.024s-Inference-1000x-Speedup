"""
CUDA operations for Softmax
"""
import math
import numpy as np
from numba import cuda

def run_softmax_cuda(x_np):
    """Run softmax operation on CUDA"""
    B, C, H, W = x_np.shape
    @cuda.jit(cache=True)
    def softmax_Cuda(inp, out):
        # tọa độ output
        b_idx = cuda.blockIdx.x  # mỗi block xử lý 1 sample trong batch
        idx   = cuda.threadIdx.x   # mỗi thread xử lý 1 trong 10 phần tử
        temp = cuda.shared.array(10, dtype=np.float32)
        temp[idx] = inp[b_idx, idx, 0, 0]
        cuda.syncthreads()
        sum_exp = 0.0
        for i in range(10):
            sum_exp +=  math.exp(temp[i])
        cuda.syncthreads() 
        out[b_idx, idx] = math.exp(temp[idx]) / sum_exp
    
    d_out = cuda.device_array((B, C), dtype=np.float32)
    bpg_x = 10
    bpg_z = B
    softmax_Cuda[B, 10](
        x_np, d_out
    )
    cuda.synchronize()
    return d_out

def run_softmax_backward(output, Y):
    """Run softmax backward operation on CUDA"""
    B, C = output.shape
    @cuda.jit(cache=True) 
    def softmax_backward(inp, y, out):
        b = cuda.blockIdx.x
        c = cuda.threadIdx.x
        out[b, c, 0, 0] = (inp[b, c] - y[b, c]) * 1/B
    
    d_out = cuda.device_array((B, C, 1, 1), dtype=np.float32)
    softmax_backward[B, C](output, Y, d_out)
    cuda.synchronize()
    return d_out
