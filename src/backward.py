"""
Backward pass implementation for SqueezeNet
"""
import math
import numpy as np
from numba import cuda
from .cuda_ops.softmax import run_softmax_cuda, run_softmax_backward
from .cuda_ops.pool import run_avgpool_backward, run_maxpooling_backward
from .cuda_ops.conv import run_conv_backward, run_filter_conv_backward, run_bias_backward
from .cuda_ops.fire_module import run_firemodule_backward

def backward_squeezenet_cuda(X_input, X, weight, Y_cuda):
    """Backward pass of SqueezeNet using CUDA operations"""
    X_output = run_softmax_cuda(X["avgpool10"])
    avgpool10_lo = run_softmax_backward(X_output, Y_cuda)
    conv10_lo = run_avgpool_backward(avgpool10_lo)
    
    fire9_lo, conv10_lw, conv10_lb = run_conv_backward(X["fire9"]["firemodule_out"],
                                                       weight["classifier.0.weight"], 
                                                       weight["classifier.0.bias"],
                                                       conv10_lo,
                                                       stride=1, padding=0, TPB=16)
    
    maxpool8_lo = run_firemodule_backward(X["maxpool8"],
                                          weight["features.12.squeeze.weight"], 
                                          weight["features.12.squeeze.bias"],
                                          X["fire9"]["squeeze"], 
                                          X["fire9"]["relu_squeeze"],
                                          weight["features.12.expand1x1.weight"], 
                                          weight["features.12.expand1x1.bias"],
                                          weight["features.12.expand3x3.weight"], 
                                          weight["features.12.expand3x3.bias"],
                                          X["fire9"]["concat"], fire9_lo)
    
    fire8_lo = run_maxpooling_backward(X["fire8"]["firemodule_out"], maxpool8_lo["LO"], K=3, stride=2)
    
    fire7_lo = run_firemodule_backward(X["fire7"]["firemodule_out"],
                                       weight["features.10.squeeze.weight"], 
                                       weight["features.10.squeeze.bias"],
                                       X["fire8"]["squeeze"], 
                                       X["fire8"]["relu_squeeze"],
                                       weight["features.10.expand1x1.weight"], 
                                       weight["features.10.expand1x1.bias"],
                                       weight["features.10.expand3x3.weight"], 
                                       weight["features.10.expand3x3.bias"],
                                       X["fire8"]["concat"], fire8_lo)
    
    fire6_lo = run_firemodule_backward(X["fire6"]["firemodule_out"],
                                       weight["features.9.squeeze.weight"], 
                                       weight["features.9.squeeze.bias"],
                                       X["fire7"]["squeeze"], 
                                       X["fire7"]["relu_squeeze"],
                                       weight["features.9.expand1x1.weight"], 
                                       weight["features.9.expand1x1.bias"],
                                       weight["features.9.expand3x3.weight"], 
                                       weight["features.9.expand3x3.bias"],
                                       X["fire7"]["concat"], fire7_lo["LO"])
    
    fire5_lo = run_firemodule_backward(X["fire5"]["firemodule_out"],
                                       weight["features.8.squeeze.weight"], 
                                       weight["features.8.squeeze.bias"],
                                       X["fire6"]["squeeze"], 
                                       X["fire6"]["relu_squeeze"],
                                       weight["features.8.expand1x1.weight"], 
                                       weight["features.8.expand1x1.bias"],
                                       weight["features.8.expand3x3.weight"], 
                                       weight["features.8.expand3x3.bias"],
                                       X["fire6"]["concat"], fire6_lo["LO"])
    
    maxpool4_lo = run_firemodule_backward(X["maxpool4"],
                                          weight["features.7.squeeze.weight"], 
                                          weight["features.7.squeeze.bias"],
                                          X["fire5"]["squeeze"], 
                                          X["fire5"]["relu_squeeze"],
                                          weight["features.7.expand1x1.weight"], 
                                          weight["features.7.expand1x1.bias"],
                                          weight["features.7.expand3x3.weight"], 
                                          weight["features.7.expand3x3.bias"],
                                          X["fire5"]["concat"], fire5_lo["LO"])
    
    fire4_lo = run_maxpooling_backward(X["fire4"]["firemodule_out"], maxpool4_lo["LO"], K=3, stride=2)
    
    fire3_lo = run_firemodule_backward(X["fire3"]["firemodule_out"],
                                       weight["features.5.squeeze.weight"], 
                                       weight["features.5.squeeze.bias"],
                                       X["fire4"]["squeeze"], 
                                       X["fire4"]["relu_squeeze"],
                                       weight["features.5.expand1x1.weight"], 
                                       weight["features.5.expand1x1.bias"],
                                       weight["features.5.expand3x3.weight"], 
                                       weight["features.5.expand3x3.bias"],
                                       X["fire4"]["concat"], fire4_lo)
    
    fire2_lo = run_firemodule_backward(X["fire2"]["firemodule_out"],
                                       weight["features.4.squeeze.weight"], 
                                       weight["features.4.squeeze.bias"],
                                       X["fire3"]["squeeze"], 
                                       X["fire3"]["relu_squeeze"],
                                       weight["features.4.expand1x1.weight"], 
                                       weight["features.4.expand1x1.bias"],
                                       weight["features.4.expand3x3.weight"], 
                                       weight["features.4.expand3x3.bias"],
                                       X["fire3"]["concat"], fire3_lo["LO"])
    
    maxpool1_lo = run_firemodule_backward(X["maxpool1"],
                                          weight["features.3.squeeze.weight"], 
                                          weight["features.3.squeeze.bias"],
                                          X["fire2"]["squeeze"], 
                                          X["fire2"]["relu_squeeze"],
                                          weight["features.3.expand1x1.weight"], 
                                          weight["features.3.expand1x1.bias"],
                                          weight["features.3.expand3x3.weight"], 
                                          weight["features.3.expand3x3.bias"],
                                          X["fire2"]["concat"], fire2_lo["LO"])
    
    conv1_relu_lo = run_maxpooling_backward(X["conv1_relu"], maxpool1_lo["LO"], K=3, stride=2)
    
    from .cuda_ops.relu import run_relu_backward
    conv1_lo = run_relu_backward(X["conv1"], conv1_relu_lo, TPB=16)
    
    bias_backward = run_bias_backward(conv1_lo, weight["features.0.bias"])
    F_backward = run_filter_conv_backward(X_input, weight["features.0.weight"], conv1_lo, 2, 3)
    
    return {
        'features.0.weight': F_backward,
        'features.0.bias': bias_backward,
        'features.3.squeeze.weight': maxpool1_lo["LF_s1x1"],
        'features.3.squeeze.bias': maxpool1_lo["LB_s1x1"],
        'features.3.expand1x1.weight': maxpool1_lo["LF_e1x1"],
        'features.3.expand1x1.bias': maxpool1_lo["LB_e1x1"],
        'features.3.expand3x3.weight': maxpool1_lo["LF_e3x3"],
        'features.3.expand3x3.bias': maxpool1_lo["LB_e3x3"],
        'features.4.squeeze.weight': fire2_lo["LF_s1x1"],
        'features.4.squeeze.bias': fire2_lo["LB_s1x1"],
        'features.4.expand1x1.weight': fire2_lo["LF_e1x1"],
        'features.4.expand1x1.bias': fire2_lo["LB_e1x1"],
        'features.4.expand3x3.weight': fire2_lo["LF_e3x3"],
        'features.4.expand3x3.bias': fire2_lo["LB_e3x3"],
        'features.5.squeeze.weight': fire3_lo["LF_s1x1"],
        'features.5.squeeze.bias': fire3_lo["LB_s1x1"],
        'features.5.expand1x1.weight': fire3_lo["LF_e1x1"],
        'features.5.expand1x1.bias': fire3_lo["LB_e1x1"],
        'features.5.expand3x3.weight': fire3_lo["LF_e3x3"],
        'features.5.expand3x3.bias': fire3_lo["LB_e3x3"],
        'features.7.squeeze.weight': maxpool4_lo["LF_s1x1"],
        'features.7.squeeze.bias': maxpool4_lo["LB_s1x1"],
        'features.7.expand1x1.weight': maxpool4_lo["LF_e1x1"],
        'features.7.expand1x1.bias': maxpool4_lo["LB_e1x1"],
        'features.7.expand3x3.weight': maxpool4_lo["LF_e3x3"],
        'features.7.expand3x3.bias': maxpool4_lo["LB_e3x3"],
        'features.8.squeeze.weight': fire5_lo["LF_s1x1"],
        'features.8.squeeze.bias': fire5_lo["LB_s1x1"],
        'features.8.expand1x1.weight': fire5_lo["LF_e1x1"],
        'features.8.expand1x1.bias': fire5_lo["LB_e1x1"],
        'features.8.expand3x3.weight': fire5_lo["LF_e3x3"],
        'features.8.expand3x3.bias': fire5_lo["LB_e3x3"],
        'features.9.squeeze.weight': fire6_lo["LF_s1x1"],
        'features.9.squeeze.bias': fire6_lo["LB_s1x1"],
        'features.9.expand1x1.weight': fire6_lo["LF_e1x1"],
        'features.9.expand1x1.bias': fire6_lo["LB_e1x1"],
        'features.9.expand3x3.weight': fire6_lo["LF_e3x3"],
        'features.9.expand3x3.bias': fire6_lo["LB_e3x3"],
        'features.10.squeeze.weight': fire7_lo["LF_s1x1"],
        'features.10.squeeze.bias': fire7_lo["LB_s1x1"],
        'features.10.expand1x1.weight': fire7_lo["LF_e1x1"],
        'features.10.expand1x1.bias': fire7_lo["LB_e1x1"],
        'features.10.expand3x3.weight': fire7_lo["LF_e3x3"],
        'features.10.expand3x3.bias': fire7_lo["LB_e3x3"],
        'features.12.squeeze.weight': maxpool8_lo["LF_s1x1"],
        'features.12.squeeze.bias': maxpool8_lo["LB_s1x1"],
        'features.12.expand1x1.weight': maxpool8_lo["LF_e1x1"],
        'features.12.expand1x1.bias': maxpool8_lo["LB_e1x1"],
        'features.12.expand3x3.weight': maxpool8_lo["LF_e3x3"],
        'features.12.expand3x3.bias': maxpool8_lo["LB_e3x3"],
        'classifier.0.weight': conv10_lw,
        'classifier.0.bias': conv10_lb
    }
