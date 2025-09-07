"""
CUDA operations for Convolution
"""
import numpy as np
from numba import cuda, float32

def run_conv2d_cuda_shared_2(x_conv, w_conv, b_conv, P=2, S=2, TPB=16):
    """Run 2D convolution with shared memory on CUDA"""
    B, C, H, W = x_conv.shape
    F, _, K, _ = w_conv.shape

    # Tính output size
    H_pad = H + 2*P
    W_pad = W + 2*P
    H_out = (H_pad - K)//S + 1
    W_out = (W_pad - K)//S + 1

    # Tính tile size cho shared memory
    tile_h = (TPB - 1) * S + K
    tile_w = tile_h
    
    @cuda.jit(cache=True)
    def conv2d_cuda_shared(inp, filters, bias, out):
        # Shared memory 3D: [C, tile_h, tile_w]
        shmem = cuda.shared.array((C, tile_h, tile_w), float32)
    
        # Tách batch và filter
        z = cuda.blockIdx.z
        b = z // F
        f = z % F
    
        # Tọa độ output
        x_out = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        y_out = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
        # Góc trên–trái của tile trên inp đã pad
        x0 = cuda.blockIdx.x * cuda.blockDim.x * S
        y0 = cuda.blockIdx.y * cuda.blockDim.y * S
        for c in range(C):
            for ly in range(cuda.threadIdx.y, tile_h, cuda.blockDim.y):
                gy = y0 + ly-P
                for lx in range(cuda.threadIdx.x, tile_w, cuda.blockDim.x):
                    gx = x0 + lx-P
                    # nếu ngoài biên input thì gán 0, còn lại đọc inp
                    if 0 <= gy < H and 0 <= gx < W:
                        shmem[c, ly, lx] = inp[b, c, gy, gx]
                    else:
                        shmem[c, ly, lx] = 0.0
        cuda.syncthreads()
    
        # 2) Tính convolution nếu trong vùng output
        if b < B and f < F and y_out < H_out and x_out < W_out:
            acc = float32(0.0)
            for c in range(C):
                for ky in range(K):
                    for kx in range(K):
                        sy = cuda.threadIdx.y * S + ky
                        sx = cuda.threadIdx.x * S + kx
                        acc += shmem[c, sy, sx] * filters[f, c, ky, kx]
            acc += bias[f]
            out[b, f, y_out, x_out] = acc
    
    d_out = cuda.device_array((B, F, H_out, W_out), dtype=np.float32)
    # grid & block dims
    bpg_x = (W_out + TPB - 1) // TPB
    bpg_y = (H_out + TPB - 1) // TPB
    bpg_z = B * F
    grid_dims = (bpg_x, bpg_y, bpg_z)
    block_dims = (TPB, TPB, 1)
    
    conv2d_cuda_shared[grid_dims, block_dims](x_conv, w_conv, b_conv, d_out)
    cuda.synchronize()
    return d_out

def run_conv2d_cuda(x, w, b, P=2, S=2, TPB=16):
    """Run 2D convolution on CUDA"""
    B, C, H, W = x.shape
    F, _, KH, KW = w.shape
    
    H_out = (H + 2*P - KH)//S + 1
    W_out = (W + 2*P - KW)//S + 1
    
    @cuda.jit(cache=True)
    def conv2d_cuda_kernel(input_arr, filters, bias, output):
        z = cuda.blockIdx.z
        b = z // F
        f = z % F
        
        x_out = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        y_out = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        
        if x_out < W_out and y_out < H_out and b < B and f < F:
            temp = float32(0.0)
            # FIXED: Loop channels first, then kernel positions
            for c in range(C):
                for kh in range(KH):
                    for kw in range(KW):
                        # Calculate input coordinates
                        y_in = y_out * S + kh - P
                        x_in = x_out * S + kw - P
                        
                        # Check bounds for each kernel position
                        if 0 <= y_in < H and 0 <= x_in < W:
                            temp += input_arr[b, c, y_in, x_in] * filters[f, c, kh, kw]
                        # If out of bounds, contribution is 0 (padding effect)
            temp += bias[f]
            output[b, f, y_out, x_out] = temp
    
    d_out = cuda.device_array((B, F, H_out, W_out), dtype=np.float32)
    bpg_x = (W_out + TPB - 1) // TPB
    bpg_y = (H_out + TPB - 1) // TPB
    bpg_z = B * F
    
    grid_dims = (bpg_x, bpg_y, bpg_z)
    block_dims = (TPB, TPB, 1)
    
    conv2d_cuda_kernel[grid_dims, block_dims](x, w, b, d_out)
    cuda.synchronize()
    
    return d_out

def run_filter_conv_backward(X, F, LO, S=1, P=1, TPB=256):
    """Run filter convolution backward on CUDA"""
    B, C, H, W = X.shape
    B_f, C_f, H_f, W_f = F.shape    
    
    @cuda.jit(cache=True)
    def filter_conv_backward(inp, fil, lo, out):
        # Mỗi thread xử lý nhiều phần tử để tận dụng tốt hơn
        tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        total_elements = B_f * C_f * H_f * W_f
        
        if tid >= total_elements:
            return
            
        # Chuyển 1D index thành 4D coordinates
        w_idx = tid % W_f
        tid //= W_f
        h_idx = tid % H_f
        tid //= H_f
        c_idx = tid % C_f
        b_idx = tid // C_f
        
        # Tính gradient cho filter[b_idx, c_idx, h_idx, w_idx]
        temp_sum = float32(0.0)
        for k in range(B):  # batch của input
            for i in range(lo.shape[2]):  # height của output gradient
                for j in range(lo.shape[3]):  # width của output gradient
                    # Tính vị trí tương ứng trong input
                    sy = i * S - P + h_idx
                    sx = j * S - P + w_idx
                    
                    # Chỉ tính nếu vị trí hợp lệ
                    if 0 <= sy < H and 0 <= sx < W:
                        temp_sum += lo[k, b_idx, i, j] * inp[k, c_idx, sy, sx]
        out[b_idx, c_idx, h_idx, w_idx] = fil[b_idx, c_idx, h_idx, w_idx] - 0.0008* temp_sum
    
    # Tạo output array
    d_out = cuda.device_array((B_f, C_f, H_f, W_f), dtype=np.float32)
    
    # Tận dụng tối đa threads
    total_threads_needed = B_f * C_f * H_f * W_f
    threads_per_block = TPB  # hoặc 512 tùy GPU
    blocks_per_grid = (total_threads_needed + TPB -1) // TPB
    
    # Launch kernel
    filter_conv_backward[blocks_per_grid, threads_per_block](X, F, LO, d_out)
    cuda.synchronize()
    
    return d_out

def run_input_conv_backward(X, F, LO, S=1, P=1, TPB=16):
    """Run input convolution backward on CUDA"""
    B, C, H, W = X.shape
    H_pad, W_pad = H + 2*P, W + 2*P
    K = F.shape[2]  # assuming F.shape = (C_out, C_in, K, K)
    B_f, C_f, H_f, W_f = F.shape
    
    @cuda.jit(cache=True)
    def input_conv_backward(filt, lo, out):
        x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # 0 -> W_pad-1
        y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y  # 0 -> H_pad-1
        z = cuda.blockIdx.z
        b = z // C
        c = z % C
        
        if x < W_pad and y < H_pad:
            temp = float32(0.0)
            for b_f in range(B_f):  # b_f is output channel
                for i in range(K):
                    for j in range(K):
                        # Calculate source coordinates in lo (output gradient)
                        # Đây là phần quan trọng: ánh xạ từ input coordinates sang output coordinates
                        if (y +i- (K-1))%S == 0 and (x +j- (K-1))%S == 0:
                            sy = (y +i- (K-1))//S  # hoặc y - (K-1-i) tùy theo cách hiểu
                            sx = (x +j- (K-1))//S  # hoặc x - (K-1-j) tùy theo cách hiểu
                            if sy >= 0 and sy < lo.shape[2] and sx >= 0 and sx < lo.shape[3]:
                                temp += lo[b, b_f, sy, sx] * filt[b_f, c, K-1-i, K-1-j]
            out[b, c, y, x] = temp 
    
    d_out = cuda.device_array((B, C, H_pad, W_pad), dtype=np.float32)
    bpg_x = (W_pad + TPB - 1) // TPB
    bpg_y = (H_pad + TPB - 1) // TPB
    bpg_z = B * C
    
    grid_dims = (bpg_x, bpg_y, bpg_z)
    block_dims = (TPB, TPB, 1)
    
    input_conv_backward[grid_dims, block_dims](F, LO, d_out)
    cuda.synchronize()
    
    return d_out[:, :, P:H_pad-P, P:W_pad-P]

def run_bias_backward(inp, bias, TPB=64):
    """Run bias backward on CUDA"""
    B, C, H, W = inp.shape
    @cuda.jit(cache=True)
    def bias1x1_backward(X, bi, d_out):
        x = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x #10
        if x < C:
            temp = float32(0.0)
            for b in range(B):
                for i in range(H):
                    for j in range(W):
                        temp += X[b, x, i, j] 
            d_out[x] = bi[x] - 0.0008 * temp
    
    d_out = cuda.device_array((C,), dtype=np.float32)
    blocksize = (C+TPB-1)//TPB
    blockdim = TPB   
    bias1x1_backward[blocksize, blockdim](inp, bias, d_out)
    cuda.synchronize()
    return d_out

def run_conv_backward(X, F, B, LO, stride=1, padding=1, TPB=16):
    """Run complete convolution backward on CUDA"""
    F_backward = run_filter_conv_backward(X, F, LO, stride, padding)
    X_backward = run_input_conv_backward(X, F, LO, stride, padding, TPB)
    bias_backward = run_bias_backward(LO, B)
    return X_backward, F_backward, bias_backward
