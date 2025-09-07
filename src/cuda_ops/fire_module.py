"""
CUDA operations for Fire Module
"""
import numpy as np
from numba import cuda
from .conv import run_conv2d_cuda
from .relu import run_Relu_Cuda
from .utils import run_concat_cuda

def fire_module_cuda(x, s_w1, s_b1, e_w1, e_b1, e_w3, e_b3):
    """Run Fire module on CUDA"""
    squeeze = run_conv2d_cuda(x, s_w1, s_b1, P=0, S=1, TPB=16)                   #8x96x55x55 ==> 8x16x55x55
    relu_squeeze = run_Relu_Cuda(squeeze, TPB=16)                            #8x16x55x55
    expand_1x1_cu = run_conv2d_cuda(relu_squeeze, e_w1, e_b1, P=0, S=1, TPB=16)    #8x16x55x55 ==> 8x64x55x55
    expand_3x3_cu = run_conv2d_cuda(relu_squeeze, e_w3, e_b3, P=1, S=1, TPB=16)   #8x16x55x55 ==> 8x64x55x55
    concat = run_concat_cuda(expand_1x1_cu, expand_3x3_cu, TPB=16)                            #8x128x55x55
    firemodule_out = run_Relu_Cuda(concat, TPB=16)
    return {
        "squeeze": squeeze,
        "relu_squeeze": relu_squeeze,
        "expand_1x1": expand_1x1_cu,
        "expand_3x3": expand_3x3_cu,
        "concat": concat,
        "firemodule_out": firemodule_out
    }

def run_firemodule_backward(X_previous_layer,
                            weights_squeeze1x1, bias_squeeze1x1,
                            X_squeeze, X_relu_squeeze,
                            weights_expand1x1, bias_expand1x1,
                            weights_expand3x3, bias_expand3x3,
                            X_concat, LO):
    """Run Fire module backward on CUDA"""
    from .relu import run_relu_backward
    from .conv import run_conv_backward
    from .utils import run_sum_matrix
    
    L_concat = run_relu_backward(X_concat, LO, TPB=16)
    C_out = L_concat.shape[1]//2
    L_e_1, L_e_3 = L_concat[:, :C_out, :, :], L_concat[:, C_out:, :, :]
    LX_1_out, LF_W_1_out, LF_B_1_out = run_conv_backward(X_relu_squeeze, weights_expand1x1, bias_expand1x1, L_e_1, stride=1, padding=0, TPB=16)
    LX_3_out, LF_W_3_out, LF_B_3_out = run_conv_backward(X_relu_squeeze, weights_expand3x3, bias_expand3x3, L_e_3, stride=1, padding=1, TPB=16)
    sum_matrix = run_sum_matrix(LX_3_out, LX_1_out, TPB=16)
    S_1x1 = run_relu_backward(X_squeeze, sum_matrix, TPB=16)
    LX_pre_out, LF_pre_out, LB_pre_out = run_conv_backward(X_previous_layer, weights_squeeze1x1, bias_squeeze1x1, S_1x1, stride=1, padding=0, TPB=16)
    return {"LO": LX_pre_out,
            "LF_s1x1": LF_pre_out,
            "LB_s1x1": LB_pre_out,
            "LF_e1x1": LF_W_1_out,
            "LB_e1x1": LF_B_1_out,
            "LF_e3x3": LF_W_3_out,
            "LB_e3x3": LF_B_3_out,
            }
