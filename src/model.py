"""
SqueezeNet Model Implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import cuda
from .cuda_ops.conv import run_conv2d_cuda_shared_2, run_conv2d_cuda
from .cuda_ops.relu import run_Relu_Cuda
from .cuda_ops.pool import run_maxpool2d_cuda, run_global_average_Cuda
from .cuda_ops.fire_module import fire_module_cuda
from .cuda_ops.softmax import run_softmax_cuda

class Fire(nn.Module):
    """Fire module for SqueezeNet"""
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super().__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)

    def forward(self, x, return_intermediates=False):
        squeeze_out = self.squeeze(x)
        relu_squeeze = F.relu(squeeze_out)
        expand1x1_out = self.expand1x1(relu_squeeze)
        expand3x3_out = self.expand3x3(relu_squeeze)
        concat = torch.cat([expand1x1_out, expand3x3_out], dim=1)
        fire_out = F.relu(concat)
        return fire_out

class SqueezeNetManual(nn.Module):
    """Manual SqueezeNet implementation with gradient tracking"""
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3),  # conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # maxpool1

            Fire(96,  16,  64,  64),   # fire2
            Fire(128, 16,  64,  64),   # fire3
            Fire(128, 32, 128, 128),   # fire4
            nn.MaxPool2d(kernel_size=3, stride=2),                  # maxpool4

            Fire(256, 32, 128, 128),   # fire5
            Fire(256, 48, 192, 192),   # fire6
            Fire(384, 48, 192, 192),   # fire7
            Fire(384, 64, 256, 256),   # fire8
            nn.MaxPool2d(kernel_size=3, stride=2),                  # maxpool8

            Fire(512, 64, 256, 256),   # fire9
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(512, num_classes, kernel_size=1),            # conv10
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # === Thêm: Lưu gradient ===
        self.gradients = {}
        self._register_gradient_hooks()

    def _register_gradient_hooks(self):
        """Đăng ký backward hook cho tất cả các tham số có requires_grad=True"""
        for name, param in self.named_parameters():
            def make_hook(name):
                def hook(grad):
                    self.gradients[name] = grad.clone()  # Lưu bản sao của gradient
                return hook

            param.register_hook(make_hook(name))

    def clear_gradients(self):
        """Xóa gradient đã lưu từ bước trước"""
        self.gradients.clear()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)

def squeezenet_forward_cuda(x_cuda, weights_cuda):
    """Forward pass of SqueezeNet using CUDA operations"""
    conv1 = run_conv2d_cuda_shared_2(x_cuda, weights_cuda['features.0.weight'], weights_cuda['features.0.bias'], P=3, S=2, TPB=16)
    conv1_relu = run_Relu_Cuda(conv1, TPB=16)
    maxpool1 = run_maxpool2d_cuda(conv1_relu, P=0, S=2, TPB=32)
    
    ########################-----Fire2----##################################
    prefix = f'features.{3}'
    sw = weights_cuda[f'{prefix}.squeeze.weight']
    sb = weights_cuda[f'{prefix}.squeeze.bias']     
    ew1 = weights_cuda[f'{prefix}.expand1x1.weight']
    eb1 = weights_cuda[f'{prefix}.expand1x1.bias']
    ew3 = weights_cuda[f'{prefix}.expand3x3.weight']
    eb3 = weights_cuda[f'{prefix}.expand3x3.bias']
    TPB = [16, 16, 16, 16, 16, 16]
    fire2 = fire_module_cuda(maxpool1, sw, sb, ew1, eb1, ew3, eb3)
    
    ########################-----Fire3----##################################
    prefix = f'features.{4}'
    sw = weights_cuda[f'{prefix}.squeeze.weight']
    sb = weights_cuda[f'{prefix}.squeeze.bias']
    ew1 = weights_cuda[f'{prefix}.expand1x1.weight']
    eb1 = weights_cuda[f'{prefix}.expand1x1.bias']
    ew3 = weights_cuda[f'{prefix}.expand3x3.weight']
    eb3 = weights_cuda[f'{prefix}.expand3x3.bias']
    TPB = [16, 16, 16, 16, 16, 16]
    fire3 = fire_module_cuda(fire2["firemodule_out"], sw, sb, ew1, eb1, ew3, eb3)
    
    ########################-----Fire4----##################################
    prefix = f'features.{5}'
    sw = weights_cuda[f'{prefix}.squeeze.weight']
    sb = weights_cuda[f'{prefix}.squeeze.bias']
    ew1 = weights_cuda[f'{prefix}.expand1x1.weight']
    eb1 = weights_cuda[f'{prefix}.expand1x1.bias']
    ew3 = weights_cuda[f'{prefix}.expand3x3.weight']
    eb3 = weights_cuda[f'{prefix}.expand3x3.bias']
    TPB = [16, 16, 16, 16, 16, 16]
    fire4 = fire_module_cuda(fire3["firemodule_out"], sw, sb, ew1, eb1, ew3, eb3)
    
    maxpool4 = run_maxpool2d_cuda(fire4["firemodule_out"], P=0, S=2, TPB=32)
    
    ########################-----Fire5----##################################
    prefix = f'features.{7}'
    sw = weights_cuda[f'{prefix}.squeeze.weight']
    sb = weights_cuda[f'{prefix}.squeeze.bias']
    ew1 = weights_cuda[f'{prefix}.expand1x1.weight']
    eb1 = weights_cuda[f'{prefix}.expand1x1.bias']
    ew3 = weights_cuda[f'{prefix}.expand3x3.weight']
    eb3 = weights_cuda[f'{prefix}.expand3x3.bias']
    TPB = [16, 16, 16, 16, 16, 16]
    fire5 = fire_module_cuda(maxpool4, sw, sb, ew1, eb1, ew3, eb3)
    
    ########################-----Fire6----##################################
    prefix = f'features.{8}'
    sw = weights_cuda[f'{prefix}.squeeze.weight']
    sb = weights_cuda[f'{prefix}.squeeze.bias']
    ew1 = weights_cuda[f'{prefix}.expand1x1.weight']
    eb1 = weights_cuda[f'{prefix}.expand1x1.bias']
    ew3 = weights_cuda[f'{prefix}.expand3x3.weight']
    eb3 = weights_cuda[f'{prefix}.expand3x3.bias']
    TPB = [16, 16, 16, 16, 16, 16]
    fire6 = fire_module_cuda(fire5["firemodule_out"], sw, sb, ew1, eb1, ew3, eb3)
    
    ########################-----Fire7----##################################
    prefix = f'features.{9}'
    sw = weights_cuda[f'{prefix}.squeeze.weight']
    sb = weights_cuda[f'{prefix}.squeeze.bias']
    ew1 = weights_cuda[f'{prefix}.expand1x1.weight']
    eb1 = weights_cuda[f'{prefix}.expand1x1.bias']
    ew3 = weights_cuda[f'{prefix}.expand3x3.weight']
    eb3 = weights_cuda[f'{prefix}.expand3x3.bias']
    TPB = [16, 16, 16, 16, 16, 16]
    fire7 = fire_module_cuda(fire6["firemodule_out"], sw, sb, ew1, eb1, ew3, eb3)
    
    ########################-----Fire8----##################################
    prefix = f'features.{10}'
    sw = weights_cuda[f'{prefix}.squeeze.weight']
    sb = weights_cuda[f'{prefix}.squeeze.bias']
    ew1 = weights_cuda[f'{prefix}.expand1x1.weight']
    eb1 = weights_cuda[f'{prefix}.expand1x1.bias']
    ew3 = weights_cuda[f'{prefix}.expand3x3.weight']
    eb3 = weights_cuda[f'{prefix}.expand3x3.bias']
    TPB = [16, 16, 16, 16, 16, 16]
    fire8 = fire_module_cuda(fire7["firemodule_out"], sw, sb, ew1, eb1, ew3, eb3)
    
    maxpool8 = run_maxpool2d_cuda(fire8["firemodule_out"], P=0, S=2, TPB=32)
    
    ########################-----Fire9----##################################
    prefix = f'features.{12}'
    sw = weights_cuda[f'{prefix}.squeeze.weight']
    sb = weights_cuda[f'{prefix}.squeeze.bias']
    ew1 = weights_cuda[f'{prefix}.expand1x1.weight']
    eb1 = weights_cuda[f'{prefix}.expand1x1.bias']
    ew3 = weights_cuda[f'{prefix}.expand3x3.weight']
    eb3 = weights_cuda[f'{prefix}.expand3x3.bias']
    TPB = [16, 16, 16, 16, 16, 16]
    fire9 = fire_module_cuda(maxpool8, sw, sb, ew1, eb1, ew3, eb3)
    
    conv10 = run_conv2d_cuda(fire9["firemodule_out"], weights_cuda['classifier.0.weight'], weights_cuda['classifier.0.bias'], P=0, S=1, TPB=16)
    avgpool10 = run_global_average_Cuda(conv10, TPB=32)
    
    return {
        'conv1': conv1,
        'conv1_relu': conv1_relu,
        'maxpool1': maxpool1,
        'fire2': fire2,
        'fire3': fire3,
        'fire4': fire4,
        'maxpool4': maxpool4,
        'fire5': fire5,
        'fire6': fire6,
        'fire7': fire7,
        'fire8': fire8,
        'maxpool8': maxpool8,
        'fire9': fire9,
        'conv10': conv10,
        'avgpool10': avgpool10,
    }
