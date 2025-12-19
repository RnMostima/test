# unified_model.py - 修复版
import torch
import torch.nn as nn

class UnifiedActionRecognizer(nn.Module):
    """
    统一动作识别模型 - 动态维度适配版
    可以处理原始视频和BDQ特征，自动适应不同的特征维度
    """
    def __init__(self, num_classes=8, feature_dim=2048, bdq_input_dim=None):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.bdq_input_dim = bdq_input_dim or feature_dim  # BDQ输入维度
        
        print(f"创建统一动作识别器: BDQ输入={self.bdq_input_dim}, 特征维度={feature_dim}, 类别数={num_classes}")
        
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
        
        # 动态维度适配器（用于BDQ输出）
        if self.bdq_input_dim == self.feature_dim:
            # 如果维度相同，使用恒等映射
            self.bdq_feature_adapter = nn.Identity()
            print(f"BDQ特征适配器: 恒等映射 ({self.bdq_input_dim} -> {self.feature_dim})")
        else:
            # 如果维度不同，使用线性层适配
            self.bdq_feature_adapter = nn.Sequential(
                nn.Linear(self.bdq_input_dim, self.feature_dim),
                nn.BatchNorm1d(self.feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            )
            print(f"BDQ特征适配器: 线性适配 ({self.bdq_input_dim} -> {self.feature_dim})")
        
        # 统一的分类器
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # 初始化权重
        self._initialize_weights()
    
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
            # BDQ路径
            if len(x.shape) > 2:
                # 如果是5D张量，展平
                x = x.view(x.size(0), -1)
            
            # 检查输入维度
            if x.shape[1] != self.bdq_input_dim:
                print(f"警告: BDQ特征维度不匹配! 期望 {self.bdq_input_dim}, 实际 {x.shape[1]}")
                # 动态调整适配器
                if hasattr(self.bdq_feature_adapter, '0') and isinstance(self.bdq_feature_adapter[0], nn.Linear):
                    # 如果是线性适配器，重建它
                    old_adapter = self.bdq_feature_adapter
                    self.bdq_input_dim = x.shape[1]
                    self.bdq_feature_adapter = nn.Sequential(
                        nn.Linear(self.bdq_input_dim, self.feature_dim),
                        nn.BatchNorm1d(self.feature_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.3)
                    ).to(x.device)
                    print(f"动态重建BDQ适配器: {self.bdq_input_dim} -> {self.feature_dim}")
            
            # 通过适配器
            x = self.bdq_feature_adapter(x)
        else:
            # 原始视频路径
            x = self.video_feature_extractor(x)
        
        # 分类
        return self.classifier(x)


class UnifiedPrivacyRecognizer(nn.Module):
    """
    统一隐私识别模型 - 修复版
    现在可以处理2D特征输入和4D图像输入
    """
    def __init__(self, num_classes=8, feature_dim=None):
        super().__init__()
        self.num_classes = num_classes
        
        # 用于处理2D特征输入的路径
        if feature_dim is not None:
            self.feature_dim = feature_dim
            print(f"隐私识别器: 特征输入模式，维度={feature_dim}")
            # 简单的特征分类器
            self.feature_classifier = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        else:
            self.feature_dim = None
            print(f"隐私识别器: 图像输入模式")
        
        # 2D卷积网络处理单帧（图像输入）
        self.image_feature_extractor = nn.Sequential(
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
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        支持2D特征输入 [B, D] 或 4D图像输入 [B, C, H, W]
        """
        # 检查输入维度
        if len(x.shape) == 2:
            # 2D特征输入
            if self.feature_dim is None:
                raise ValueError("模型未初始化为特征输入模式，但接收到2D输入")
            
            if x.shape[1] != self.feature_dim:
                print(f"[警告] 输入特征维度{x.shape[1]}与期望{self.feature_dim}不匹配")
                # 动态调整分类器
                if hasattr(self, 'feature_classifier'):
                    old_classifier = self.feature_classifier
                    input_dim = x.shape[1]
                    self.feature_classifier = nn.Sequential(
                        nn.Linear(input_dim, 256),
                        nn.BatchNorm1d(256),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.3),
                        nn.Linear(256, self.num_classes)
                    ).to(x.device)
                    self.feature_dim = input_dim
                    print(f"动态调整特征分类器: {input_dim} -> 256 -> {self.num_classes}")
            
            return self.feature_classifier(x)
        
        elif len(x.shape) == 4:
            # 4D图像输入 [B, C, H, W]
            return self.image_feature_extractor(x)
        
        elif len(x.shape) == 5:
            # 5D视频输入 [B, C, T, H, W]，取中间帧
            T = x.shape[2]
            x = x[:, :, T//2, :, :]  # 取时间维度的中间帧
            return self.image_feature_extractor(x)
        
        else:
            raise ValueError(f"不支持的输入维度: {x.shape}")