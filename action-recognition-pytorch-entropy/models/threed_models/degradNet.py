# degradNet.py - 简化修复版
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import numbers  # 添加numbers模块导入

class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma, dim=3):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [1, kernel_size, kernel_size]
        if isinstance(sigma, numbers.Number):
            sigma = [1, sigma, sigma]

        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.groups = channels
        self.k_size = kernel_size[-1]

        self.conv = nn.Conv3d(3, 3, groups=self.groups, kernel_size=kernel_size, 
                              bias=False, padding=(0, (self.k_size-1)//2, (self.k_size-1)//2))
        self.conv.weight = torch.nn.Parameter(torch.FloatTensor(kernel))
        self.conv.weight.requires_grad = False

    def forward(self, input):
        input = self.conv(input)
        self.conv.weight.data = torch.clamp(self.conv.weight.data, min=0)
        return input

class ResNet(nn.Module):
    def __init__(self, kernel_size=11, sig_scale=5, num_filters=4, 
                 kernel_path=None, learn_noise=True, mxp=False,
                 quantize_bits=4, quantize=True, avg=False, mode='bilinear', 
                 red_dim=16, output_dim=2048):
        super(ResNet, self).__init__()
        
        self.gauss = GaussianSmoothing(3, 5, 3)
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.bits = quantize_bits
        self.mode = mode
        self.red_dim = red_dim
        self.sig_scale = sig_scale
        self.output_dim = output_dim
        
        # 直接输出2048维的特征提取器
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 7, 7)),
            nn.Conv3d(3, 32, kernel_size=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )
        
        self.bias = nn.Parameter(
            torch.from_numpy(np.linspace(0, 2**self.bits-1, 2**self.bits, dtype='float32')[:-1] + 0.5)
        )
        self.levels = np.linspace(0, 2**self.bits-1, 2**self.bits, dtype='float32')[:-1]
        
        print(f"BDQ编码器: 输出维度 = {output_dim}")
    
    def forward(self, x):
        # 高斯平滑
        x = self.gauss(x)
        
        # 帧差计算
        x_roll = torch.roll(x, 1, dims=2)
        x = x - x_roll
        x = x[:, :, 1:, :, :]
        
        # 量化处理
        qmin = 0.
        qmax = 2. ** self.bits - 1.
        min_value = x.min()
        max_value = x.max()
        scale_value = (max_value - min_value) / (qmax - qmin)
        scale_value = max(scale_value, 1e-4)
        
        x = ((x - min_value) / ((max_value - min_value) + 1e-4)) * (qmax - qmin)
        
        y = torch.zeros(x.shape, device=x.device)
        self.bias.data = self.bias.data.clamp(0, (2 ** self.bits - 1))
        self.bias.data = self.bias.data.sort(0).values
        
        for i in range(self.levels.shape[0]):
            y = y + torch.sigmoid(self.sig_scale * (x - self.bias[i]))
        
        y = y.mul(scale_value).add(min_value)
        
        # 特征提取
        features = self.feature_extractor(y)
    
    # 添加调试信息
        if not hasattr(self, '_debug_printed'):
          print(f"[degradNet] 特征提取器输出形状: {features.shape}")
          self._debug_printed = True
    
        return features, self.bias

def resnet_degrad():
    return ResNet(output_dim=2048)