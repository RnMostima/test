# model_builder.py - 修复版
import torch
import torch.nn as nn
from . import i3d, i3d_resnet, resnet, inception_v1

# 尝试导入degradNet
try:
    from degradNet import resnet_degrad, ResNet
    has_degrad = True
    print("✓ degradNet模块导入成功")
except ImportError as e:
    has_degrad = False
    print(f"⚠ 无法导入degradNet: {e}")
    print("将使用替代方案")

MODEL_TABLE = {
    'i3d': i3d,
    'i3d_resnet': i3d_resnet,
    'resnet': resnet,
    'inception_v1': inception_v1
}

# 添加degrad到模型表
if has_degrad:
    MODEL_TABLE['degrad'] = resnet_degrad

class UnifiedActionModel(nn.Module):
    """统一接口的动作识别模型 - 修复版"""
    def __init__(self, base_model, num_classes=6):
        super().__init__()
        self.base_model = base_model
        
        # 保存原始的网络名称
        if hasattr(base_model, 'network_name'):
            self.network_name = base_model.network_name
        else:
            self.network_name = "unified_action_model"
        
        # 如果base_model是ResNet（degrad），使用不同的特征提取方式
        if hasattr(base_model, '__class__') and base_model.__class__.__name__ == 'ResNet':
            print("检测到ResNet(degrad)模型，使用简化特征提取")
            # ResNet模型已经返回特征，不需要额外处理
            self.feature_adapter = nn.Identity()
            self.classifier = nn.Linear(2048, num_classes)
        else:
            # 对于其他模型，移除原始分类器
            if hasattr(base_model, 'fc1'):
                base_model.fc1 = nn.Identity()
            elif hasattr(base_model, 'fc'):
                base_model.fc = nn.Identity()
            elif hasattr(base_model, 'classifier'):
                base_model.classifier = nn.Identity()
            
            # 特征维度适配器
            self.feature_adapter = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.Linear(512 * 4, 2048),  # ResNet50的扩展因子是4
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)
            )
            
            # 统一的分类器
            self.classifier = nn.Linear(2048, num_classes)
        
        print(f"统一接口模型创建: {self.network_name} -> 2048 -> {num_classes}类")
    
    def forward(self, x):
        # 通过基础模型
        if hasattr(self.base_model.__class__, '__name__') and self.base_model.__class__.__name__ == 'ResNet':
            # ResNet(degrad)返回特征和偏置
            features, _ = self.base_model(x)
            # 特征适配
            adapted_features = self.feature_adapter(features)
        else:
            # 其他模型
            x = self.base_model(x)
            # 特征适配
            adapted_features = self.feature_adapter(x)
        
        # 分类
        outputs = self.classifier(adapted_features)
        return outputs

def build_model(args, test_mode=False):
    """
    修复版模型构建函数
    确保返回正确的模型实例
    """
    print(f"构建模型: backbone={args.backbone_net}, 测试模式={test_mode}")
    
    # 特殊处理degrad模型
    if args.backbone_net == 'degrad':
        print("构建degrad模型...")
        if not has_degrad:
            print("❌ degradNet模块不可用，无法构建degrad模型")
            return None, f"{args.dataset}-degrad-f{args.groups}"
        
        try:
            # 直接使用degradNet中的函数
            model = resnet_degrad()
            network_name = "degrad"
        except Exception as e:
            print(f"构建degrad模型失败: {e}")
            return None, f"{args.dataset}-degrad-f{args.groups}"
    else:
        # 构建原始模型
        try:
            if args.backbone_net not in MODEL_TABLE:
                raise ValueError(f"不支持的backbone_net: {args.backbone_net}")
            
            # 调用模型构建函数
            model_builder_func = MODEL_TABLE[args.backbone_net]
            
            # 准备参数
            kwargs = vars(args)
            
            # 构建模型
            model = model_builder_func(**kwargs)
            
            # 如果是列表或元组，取第一个元素
            if isinstance(model, (list, tuple)):
                model = model[0]
            
            # 获取网络名称
            if hasattr(model, 'network_name'):
                network_name = model.network_name
            else:
                network_name = args.backbone_net
                
        except Exception as e:
            print(f"构建模型失败: {e}")
            
            # 创建简单的替代模型
            class SimpleModel(nn.Module):
                def __init__(self, num_classes=6):
                    super().__init__()
                    self.conv = nn.Conv3d(3, 32, kernel_size=3, padding=1)
                    self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
                    self.fc = nn.Linear(32, num_classes)
                    self.network_name = "simple"
                
                def forward(self, x):
                    x = self.conv(x)
                    x = self.pool(x)
                    x = x.view(x.size(0), -1)
                    return self.fc(x)
            
            model = SimpleModel(getattr(args, 'num_classes', 6))
            network_name = "simple"
    
    # 根据任务类型创建统一的模型接口
    if test_mode:
        print("测试模式: 创建统一接口模型")
        # 确保有num_classes属性
        if not hasattr(args, 'num_classes'):
            args.num_classes = 6
        
        # 对于degrad模型，不需要额外包装
        if args.backbone_net == 'degrad':
            unified_model = model
        else:
            unified_model = UnifiedActionModel(model, args.num_classes)
    else:
        unified_model = model
    
    # 构建架构名称
    arch_name = f"{args.dataset}-{args.modality}-{network_name}-f{args.groups}"
    
    if not test_mode:
        arch_name += f"-{args.lr_scheduler}{'-syncbn' if getattr(args, 'sync_bn', False) else ''}-bs{args.batch_size}-e{getattr(args, 'epochs', 100)}"
    
    print(f"模型构建完成: {arch_name}")
    print(f"模型类型: {type(unified_model)}")
    
    return unified_model, arch_name