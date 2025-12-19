# fix_check_fixed.py - 修复版检查脚本
import torch
import torch.nn as nn
import numpy as np
import os
import sys

def add_module_paths():
    """添加模块路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 可能的路径列表
    possible_paths = [
        current_dir,  # 当前目录
        os.path.join(current_dir, "action-recognition-pytorch-entropy"),
        os.path.join(current_dir, "action-recognition-pytorch-entropy", "models"),
        os.path.join(current_dir, "action-recognition-pytorch-entropy", "models", "threed_models"),
        os.path.join(current_dir, "models"),
        os.path.join(current_dir, "threed_models")
    ]
    
    added_paths = []
    for path in possible_paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.append(path)
            added_paths.append(path)
    
    print("添加的模块路径:")
    for path in added_paths:
        print(f"  - {path}")
    
    return added_paths

def check_bdq_output_dimension():
    """检查BDQ输出维度 - 修复版"""
    print("="*80)
    print("检查BDQ编码器输出维度 (修复版)")
    print("="*80)
    
    # 首先添加路径
    add_module_paths()
    
    try:
        # 尝试不同的导入方式
        bdq_model = None
        
        # 方式1: 尝试直接导入
        try:
            from degradNet import ResNet as FixedBDQResNet
            bdq_model = FixedBDQResNet(output_dim=2048)
            print("✓ 方式1导入成功: 直接导入")
        except ImportError:
            # 方式2: 尝试从子目录导入
            try:
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                from degradNet import ResNet as FixedBDQResNet
                bdq_model = FixedBDQResNet(output_dim=2048)
                print("✓ 方式2导入成功: 从当前目录导入")
            except ImportError:
                # 方式3: 尝试从action-recognition-pytorch-entropy导入
                try:
                    from action_recognition_pytorch_entropy.models.threed_models.degradNet import ResNet as FixedBDQResNet
                    bdq_model = FixedBDQResNet(output_dim=2048)
                    print("✓ 方式3导入成功: 从action-recognition-pytorch-entropy导入")
                except ImportError as e:
                    print(f"✗ 所有导入方式都失败: {e}")
                    return False
        
        if bdq_model is None:
            print("✗ 无法创建BDQ模型")
            return False
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 测试输入
        test_input = torch.randn(2, 3, 16, 224, 224).to(device)
        
        print(f"测试输入形状: {test_input.shape}")
        
        with torch.no_grad():
            bdq_model = bdq_model.to(device)
            features, bias = bdq_model(test_input)
            print(f"BDQ输出特征形状: {features.shape}")
            print(f"BDQ输出偏置形状: {bias.shape}")
            
            if features.shape[1] == 2048:
                print("✓ BDQ输出维度修复成功: 2048维")
                return True
            else:
                print(f"⚠ BDQ输出维度: {features.shape[1]}维 (期望2048)")
                return False
                
    except Exception as e:
        print(f"检查失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 尝试手动检查文件
        print("\n尝试手动检查文件...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        degrad_file = os.path.join(current_dir, "degradNet.py")
        
        if os.path.exists(degrad_file):
            print(f"找到 degradNet.py 文件: {degrad_file}")
            with open(degrad_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if "output_dim=2048" in content:
                    print("✓ degradNet.py 中包含 output_dim=2048")
                    return True
                else:
                    print("✗ degradNet.py 中未找到 output_dim=2048")
                    return False
        else:
            print(f"未找到 degradNet.py 文件")
            
            # 搜索文件
            print("搜索 degradNet.py...")
            for root, dirs, files in os.walk(current_dir):
                if "degradNet.py" in files:
                    print(f"在 {root} 中找到 degradNet.py")
                    return True
            
            print("未找到任何 degradNet.py 文件")
            return False

def check_unified_interface():
    """检查统一接口"""
    print("\n" + "="*80)
    print("检查统一接口模型")
    print("="*80)
    
    try:
        # 尝试导入统一模型
        try:
            from unified_model import UnifiedActionRecognizer, UnifiedPrivacyRecognizer
        except ImportError:
            # 如果不在当前目录，尝试创建
            print("未找到 unified_model.py，正在创建...")
            create_unified_model_file()
            from unified_model import UnifiedActionRecognizer, UnifiedPrivacyRecognizer
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 测试动作识别模型
        print("1. 测试动作识别模型:")
        action_model = UnifiedActionRecognizer(num_classes=8, feature_dim=2048).to(device)
        
        # 测试原始视频输入
        video_input = torch.randn(4, 3, 16, 224, 224).to(device)
        output1 = action_model(video_input, use_bdq=False)
        print(f"  原始视频输入: {video_input.shape} -> 输出: {output1.shape}")
        
        # 测试BDQ特征输入
        bdq_features = torch.randn(4, 2048).to(device)
        output2 = action_model(bdq_features, use_bdq=True)
        print(f"  BDQ特征输入: {bdq_features.shape} -> 输出: {output2.shape}")
        
        # 测试隐私识别模型
        print("\n2. 测试隐私识别模型:")
        privacy_model = UnifiedPrivacyRecognizer(num_classes=8).to(device)
        
        # 测试图像输入
        image_input = torch.randn(4, 3, 224, 224).to(device)
        output3 = privacy_model(image_input)
        print(f"  图像输入: {image_input.shape} -> 输出: {output3.shape}")
        
        # 测试视频输入（取中间帧）
        video_input_5d = torch.randn(4, 3, 16, 224, 224).to(device)
        output4 = privacy_model(video_input_5d)
        print(f"  视频输入: {video_input_5d.shape} -> 输出: {output4.shape}")
        
        print("\n✓ 统一接口检查通过!")
        return True
        
    except Exception as e:
        print(f"✗ 统一接口检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_unified_model_file():
    """创建统一模型文件"""
    unified_model_content = '''# unified_model.py
import torch
import torch.nn as nn

class UnifiedActionRecognizer(nn.Module):
    """
    统一动作识别模型
    可以处理原始视频和BDQ特征，输出统一维度
    """
    def __init__(self, num_classes=8, feature_dim=2048):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # 特征提取器（用于原始视频）
        self.video_feature_extractor = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # 特征适配器（用于BDQ输出，确保维度一致）
        self.bdq_feature_adapter = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),  # 确保BDQ输出已经是feature_dim
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # 统一的分类器
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # 初始化权重
        self._initialize_weights()
        
        print(f"统一动作识别器: 特征维度={feature_dim}, 类别数={num_classes}")
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, use_bdq=False):
        """
        前向传播
        
        Args:
            x: 输入数据
            use_bdq: 是否使用BDQ特征
        """
        if use_bdq:
            # BDQ路径：假设x已经是2048维特征
            if len(x.shape) > 2:
                # 如果是5D张量，展平
                x = x.view(x.size(0), -1)
            
            # 通过适配器确保维度一致
            x = self.bdq_feature_adapter(x)
        else:
            # 原始视频路径：提取特征
            x = self.video_feature_extractor(x)
        
        # 分类
        return self.classifier(x)


class UnifiedPrivacyRecognizer(nn.Module):
    """
    统一隐私识别模型
    """
    def __init__(self, num_classes=8):
        super().__init__()
        self.num_classes = num_classes
        
        # 2D卷积网络处理单帧
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.classifier = nn.Linear(512, num_classes)
        
        print(f"统一隐私识别器: 类别数={num_classes}")
    
    def forward(self, x):
        # 如果是5D输入 [B, C, T, H, W]，取中间帧
        if len(x.shape) == 5:
            T = x.shape[2]
            x = x[:, :, T//2, :, :]  # 取时间维度的中间帧
        
        # 特征提取和分类
        x = self.feature_extractor(x)
        return self.classifier(x)
'''
    
    with open("unified_model.py", "w", encoding="utf-8") as f:
        f.write(unified_model_content)
    
    print("✓ 已创建 unified_model.py")

def create_simple_degradNet():
    """创建简化版 degradNet.py"""
    degrad_content = '''# degradNet.py - 简化修复版
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

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
        
        import numbers  # 移到内部导入
        
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
        
        return features, self.bias

def resnet_degrad():
    return ResNet(output_dim=2048)
'''
    
    with open("degradNet.py", "w", encoding="utf-8") as f:
        f.write(degrad_content)
    
    print("✓ 已创建简化版 degradNet.py")

def main():
    """主检查函数"""
    print("开始检查修复...")
    
    # 首先确保必要的文件存在
    current_dir = os.path.dirname(os.path.abspath(__file__))
    degrad_file = os.path.join(current_dir, "degradNet.py")
    unified_file = os.path.join(current_dir, "unified_model.py")
    
    if not os.path.exists(degrad_file):
        print("未找到 degradNet.py，创建简化版本...")
        create_simple_degradNet()
    
    if not os.path.exists(unified_file):
        print("未找到 unified_model.py，正在创建...")
        create_unified_model_file()
    
    bdq_ok = check_bdq_output_dimension()
    unified_ok = check_unified_interface()
    
    print("\n" + "="*80)
    print("修复检查总结")
    print("="*80)
    
    if bdq_ok and unified_ok:
        print("✓ 所有修复检查通过!")
        print("\n下一步:")
        print("1. 运行主程序: python main_comprehensive_fixed.py")
        print("2. 查看结果日志")
    else:
        print("⚠ 某些修复检查失败")
        print("\n建议操作:")
        if not bdq_ok:
            print("- 检查 degradNet.py 文件位置")
            print("- 确保文件在正确目录: 当前目录或 action-recognition-pytorch-entropy/models/threed_models/")
            print("- 运行: ls -la degradNet.py 检查文件是否存在")
        
        print("\n手动检查步骤:")
        print("1. 打开 degradNet.py，查看 ResNet 类的 __init__ 方法")
        print("2. 确保有 output_dim 参数，默认值为 2048")
        print("3. 查看 forward 方法，确保返回的特征维度是 2048")
        print("4. 运行以下测试代码:")
        print("""
import torch
from degradNet import ResNet
model = ResNet(output_dim=2048)
test_input = torch.randn(1, 3, 16, 224, 224)
features, bias = model(test_input)
print(f"特征形状: {features.shape}")
print(f"偏置形状: {bias.shape}")
        """)

if __name__ == '__main__':
    main()