"""
CUDA operations for Pooling
"""
import numpy as np
from numba import cuda

def run_global_average_Cuda(x, TPB=32):
    """Run global average pooling on CUDA"""
    B, C, H, W = x.shape
    @cuda.jit(cache=True)
    def Global_average_Cuda(inp, out):
        x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if x < B*C:
            b = x // C
            c = x % C
            acc = 0
            for i in range(H):
                for j in range(W):
                    acc += inp[b, c, i, j]
            out[b, c, 0, 0] = acc/(H*W)
    
    griddim = (B * C + TPB - 1)//TPB
    d_out = cuda.device_array((B, C, 1, 1), dtype=np.float32)
    Global_average_Cuda[griddim, TPB](
        x, d_out
    )
    cuda.synchronize()
    return d_out

def run_avgpool_backward(inp, TPB=16):
    """Run average pooling backward on CUDA"""
    B, C, _, _ = inp.shape
    H_out, W_out = 13, 13
    @cuda.jit(cache=True)
    def avgpool_backward(inp_1, out):
        z = cuda.blockIdx.z
        b = z // C
        c = z % C
        x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        if x < H_out and y < W_out:
           out[b, c, y, x] = inp_1[b, c, 0, 0]/(H_out*W_out) 
    
    d_out = cuda.device_array((B, C, H_out, W_out), dtype=np.float32)
    bpx = (H_out + TPB -1)//TPB
    bpy = (W_out + TPB -1)//TPB
    bpz = B*C
    grid_dim = (bpx, bpy, bpz)
    block_dim = (TPB, TPB, 1)
    avgpool_backward[grid_dim, block_dim](inp, d_out)
    cuda.synchronize()
    return d_out

def run_maxpool2d_cuda(x, P=0, S=2, TPB=16):
    """Run max pooling 2D on CUDA"""
    B, C, H, W = x.shape
    K = 3    
    # Tính kích thước đầu ra
    H_out = (H - K)//S + 1
    W_out = (W - K)//S + 1
    
    @cuda.jit(cache=True)
    def maxpool2d_cuda(inp, out):
        # merge batch và channel vào blockIdx.z
        z = cuda.blockIdx.z
        b = z // C
        c = z % C
    
        # tọa độ output
        x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
        if b < B and c < C and y < H_out and x < W_out:
            # gốc cửa sổ pooling
            base_y = y * S
            base_x = x * S
    
            # khởi tạo max từ phần tử đầu
            max_val = inp[b, c, base_y, base_x]
            # quét cửa sổ K×K
            for i in range(K):
                for j in range(K):
                    val = inp[b, c, base_y + i, base_x + j]
                    if val > max_val:
                        max_val = val
    
            out[b, c, y, x] = max_val
    
    # Chuẩn bị output trên GPU
    d_out = cuda.device_array((B, C, H_out, W_out), dtype=np.float32)
    
    bpg_x = (W_out + TPB - 1) // TPB
    bpg_y = (H_out + TPB - 1) // TPB
    bpg_z = B * C
    
    grid_dims = (bpg_x, bpg_y, bpg_z)
    block_dims = (TPB, TPB, 1)
    maxpool2d_cuda[grid_dims, block_dims](
        x, d_out
    )
    cuda.synchronize()
    return d_out

def run_maxpooling_backward(X, LO, K=3, stride=2, TPB=16):
    """Run max pooling backward on CUDA"""
    B, C, H, W = X.shape
    B_lo, C_lo, H_lo, W_lo = LO.shape
    
    @cuda.jit(cache=True)
    def maxpooling_backward(inp, lo, out):
        x = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
        y = cuda.blockIdx.y*cuda.blockDim.y + cuda.threadIdx.y
        z = cuda.blockIdx.z
        b = z//C
        c = z%C
        if x < W_lo and y < H_lo:
            basex = x*stride
            basey = y*stride
            maxval = -1e20
            for i in range(K):
                for j in range(K):
                    val = inp[b, c, basey + i, basex + j]
                    if val > maxval:
                        maxval = val
                        index_y = basey + i
                        index_x = basex + j
            cuda.atomic.add(out, (b, c, index_y, index_x), lo[b, c, y, x])           
    
    dout = cuda.to_device(np.zeros((B, C, H, W), dtype=np.float32))
    bpx = (W_lo + TPB -1)//TPB
    bpy = (H_lo + TPB -1)//TPB
    bpz = B_lo * C_lo
    griddim = (bpx, bpy, bpz)
    blockdim = (TPB, TPB, 1)
    maxpooling_backward[griddim, blockdim](X, LO, dout)
    cuda.synchronize()
    return dout
