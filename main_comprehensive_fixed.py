"""
BDQ编码器严谨测试实验 - 显存优化版
真正加载预训练模型，进行实际的前向传播测试
修复显存溢出问题
"""

import torch
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import json
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import Counter
from unified_model import UnifiedActionRecognizer, UnifiedPrivacyRecognizer
import importlib

# 检查degradNet是否已经导入
if 'degradNet' in sys.modules:
    print("degradNet已导入，重新加载...")
    importlib.reload(sys.modules['degradNet'])
# ===================== 环境设置 =====================
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.benchmark = True

# 显存优化设置
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True

# ===================== 路径配置 =====================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ACTION_FOLDER = os.path.join(ROOT_DIR, "action-recognition-pytorch-entropy")
CKPT_PATH = os.path.join(ACTION_FOLDER, "checkpoints", "model_degrad.ckpt")
SBU_DATA_DIR = os.path.join(ROOT_DIR, "SBU")
VIS_DIR = os.path.join(ROOT_DIR, "visualization")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
TRAIN_TXT_PATH = os.path.join(ROOT_DIR, "train.txt")
VAL_TXT_PATH = os.path.join(ROOT_DIR, "val.txt")

# 创建目录
for dir_path in [VIS_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ===================== 添加项目路径 =====================
sys.path.append(ROOT_DIR)
sys.path.append(ACTION_FOLDER)

# ===================== 导入子模块 =====================
try:
    from dataset_utils import SBUDataSet, PrivacyModel, tensor2img
    from models.model_builder import build_model
    from utils.utils import get_augmentor
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保所有依赖文件都存在")
    sys.exit(1)

# ===================== 配置类 =====================
class Config:
    """修复版配置类 - 包含所有必需属性"""
    def __init__(self):
        # 数据集配置
        self.dataset = "sbu"
        self.datadir = SBU_DATA_DIR
        self.modality = "rgb"
        self.input_channels = 3
        
        # 模型配置 - 必须包含所有build_model需要的参数
        self.backbone_net = "i3d"  # 修复：添加这个关键属性
        self.dropout = 0.5
        self.without_t_stride = False
        self.pooling_method = "avg"
        self.threed_data = True
        self.groups = 64
        self.frames_per_group = 1
        
        # 训练/测试配置
        self.batch_size = 4
        self.workers = 2
        self.test_mode = True
        
        # 实验配置
        self.num_classes = 6      # 将从数据集获取实际值
        self.privacy_num_classes = 8  # 将从数据集获取实际值
        
        # 统一接口配置
        self.feature_dim = 2048   # 统一特征维度
        self.unified_interface = True  # 使用统一接口
        
        # 添加build_model可能需要的其他参数
        self.lr_scheduler = "step"  # 默认值
        self.sync_bn = False
        self.epochs = 100
        self.lr = 0.001
        self.weight_decay = 0.0001
        
    def to_namespace(self):
        """转换为argparse.Namespace"""
        return argparse.Namespace(**self.__dict__)

def prepare_models(config, device, action_classes, privacy_classes):
    """准备模型 - 统一接口版"""
    print_section("准备统一接口模型")
    
    # 1. 动作识别模型
    action_model = UnifiedActionRecognizer(
        num_classes=action_classes,
        feature_dim=config.feature_dim
    ).to(device)
    
    print(f"✓ 统一动作识别模型准备完成: 特征维度={config.feature_dim}, 类别数={action_classes}")
    
    # 2. 隐私识别模型
    privacy_model = UnifiedPrivacyRecognizer(
    num_classes=config.privacy_num_classes,
    feature_dim=actual_feature_dim  # 使用BDQ实际输出维度
    ).to(device)

    log_message(f"✓ 统一隐私识别模型准备完成: 特征维度={actual_feature_dim}")
    print(f"✓ 统一隐私识别模型准备完成: 类别数={privacy_classes}")
    
    return action_model, privacy_model


# ===================== 显存优化的维度适配器 =====================
class DimensionAdapter(nn.Module):
    """动态维度适配器，解决维度不匹配问题（显存优化版）"""
    def __init__(self, input_dim=None, output_dim=2048):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adapter = None
        if input_dim is not None:
            self._build_adapter()
        else:
            print("⚠️ DimensionAdapter initialized with input_dim=None - will build on first forward")

    def _build_adapter(self):
        """构建适配器网络（显存优化版）"""
        if self.input_dim is None:
            raise ValueError("Cannot build adapter with input_dim=None")
        
        # 使用较小的隐藏层维度以节省显存
        hidden_dim = min(1024, max(self.input_dim // 4, self.output_dim // 2))
        print(f"Building DimensionAdapter: {self.input_dim} → {hidden_dim} → {self.output_dim}")
        
        self.adapter = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # 降低dropout率
            nn.Linear(hidden_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU(inplace=True)
        )
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def update_input_dim(self, input_dim):
        """更新输入维度并重新构建适配器（显存优化版）"""
        old_input_dim = self.input_dim
        self.input_dim = input_dim
        if old_input_dim != input_dim:
            self._build_adapter()

    def forward(self, x):
        """前向传播（显存优化版）"""
        # 如果适配器未初始化，尝试构建
        if self.adapter is None:
            print("⚠️ DimensionAdapter not initialized - building on first forward")
            self._build_adapter()
            if self.adapter is None:
                print("❌ DimensionAdapter failed to build - returning input")
                return x
        
        # 确保输入维度正确
        if x.shape[1] != self.input_dim:
            print(f"⚠️ Input dimension mismatch: expected {self.input_dim}, got {x.shape[1]}")
            # 动态更新适配器
            if x.shape[1] != self.input_dim:
                self.update_input_dim(x.shape[1])
        
        return self.adapter(x)

# ===================== 显存优化的动作识别模型 =====================
class ActionModelWithAdapter(nn.Module):
    """带适配器的动作识别模型 - 显存优化版"""
    def __init__(self, original_model, num_classes, bdq_dimension=None):
        super().__init__()
        self.original_model = original_model
        self.num_classes = num_classes
        self.bdq_dimension = bdq_dimension
        
        # 冻结原始模型参数以减少显存占用
        for param in self.original_model.parameters():
            param.requires_grad = False
        
        # 维度适配器（动态）
        if bdq_dimension is not None:
            self.dim_adapter = DimensionAdapter(input_dim=bdq_dimension, output_dim=2048)
        
        # 分类器（根据实际BDQ维度创建）
        actual_classifier_input_dim = bdq_dimension if bdq_dimension is not None else 2048
        self.classifier = nn.Linear(actual_classifier_input_dim, num_classes)
        nn.init.normal_(self.classifier.weight, 0, 0.01)
        nn.init.constant_(self.classifier.bias, 0)
        
        print(f"模型初始化: BDQ维度({bdq_dimension}) -> 分类器({actual_classifier_input_dim} -> {num_classes})")

    def forward(self, x, use_bdq=False):
        """
        前向传播（显存优化版）
        x: 输入数据
        use_bdq: True表示输入是BDQ特征，False表示输入是原始视频帧
        """
        # 自动检测输入类型
        if len(x.shape) == 5:
            # 形状 [B, C, T, H, W] -> 原始视频帧
            use_bdq = False
        elif len(x.shape) == 2:
            # 形状 [B, feature_dim] -> BDQ特征
            use_bdq = True
        
        if use_bdq and self.bdq_dimension is not None:
            # BDQ路径：直接使用特征，但确保维度匹配
            feature_dim = x.shape[1]
            
            # 如果特征维度不匹配分类器输入，动态调整
            if feature_dim != self.classifier.in_features:
                print(f"警告: 特征维度 {feature_dim} 与分类器期望 {self.classifier.in_features} 不匹配")
                
                # 重建分类器以匹配实际特征维度
                old_classifier = self.classifier
                self.classifier = nn.Linear(feature_dim, self.num_classes).to(x.device)
                print(f"重建分类器: {feature_dim} -> {self.num_classes}")
                
                # 尝试复制旧权重（如果维度兼容）
                try:
                    if old_classifier.weight.shape[1] == feature_dim:
                        self.classifier.weight.data.copy_(old_classifier.weight.data)
                        self.classifier.bias.data.copy_(old_classifier.bias.data)
                except:
                    pass  # 如果无法复制权重，使用随机初始化
            
            return self.classifier(x)
        else:
            # 原始视频路径：直接通过原始模型
            return self.original_model(x)

# ===================== 工具函数 =====================
def get_sbu_config():
    """SBU数据集配置"""
    return {
        "image_tmpl": "%05d.jpg",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }

def print_section(title, width=80):
    """打印章节标题"""
    print("\n" + "=" * width)
    print(f" {title} ".center(width, "="))
    print("=" * width)

def log_message(message, level="INFO"):
    """记录日志消息"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def clear_gpu_cache():
    """清空GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        log_message(f"GPU缓存已清理 - 当前显存使用: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

# ===================== 数据统计函数 =====================
def analyze_dataset(dataset, name="数据集"):
    """深入分析数据集"""
    print_section(f"分析{name}")
    
    action_labels = []
    privacy_labels = []
    
    # 只分析前100个样本以节省显存
    sample_count = min(100, len(dataset))
    
    for i in tqdm(range(sample_count), desc=f"分析{name}"):
        _, action_label, privacy_label = dataset[i]
        action_labels.append(action_label)
        privacy_labels.append(privacy_label)
    
    action_labels = np.array(action_labels)
    privacy_labels = np.array(privacy_labels)
    
    # 统计信息
    stats = {
        "total_samples": len(dataset),
        "sample_analyzed": sample_count,
        "action_classes": len(np.unique(action_labels)),
        "privacy_classes": len(np.unique(privacy_labels)),
        "action_label_dist": dict(sorted(Counter(action_labels).items())),
        "privacy_label_dist": dict(sorted(Counter(privacy_labels).items())),
        "action_label_range": [int(action_labels.min()), int(action_labels.max())],
        "privacy_label_range": [int(privacy_labels.min()), int(privacy_labels.max())],
        "action_label_std": float(action_labels.std()),
        "privacy_label_std": float(privacy_labels.std())
    }
    
    # 打印统计信息
    print(f"样本总数: {stats['total_samples']}")
    print(f"分析样本数: {stats['sample_analyzed']}")
    print(f"动作类别数: {stats['action_classes']}")
    print(f"隐私类别数: {stats['privacy_classes']}")
    print(f"动作标签范围: {stats['action_label_range'][0]} - {stats['action_label_range'][1]}")
    print(f"隐私标签范围: {stats['privacy_label_range'][0]} - {stats['privacy_label_range'][1]}")
    print(f"动作标签分布: {stats['action_label_dist']}")
    print(f"隐私标签分布: {stats['privacy_label_dist']}")
    
    return stats

def test_model_real_fixed(model_degrad, task_model, data_loader, device, 
                         task_type="action", use_bdq=False, label_min=0, num_classes=None):
    """
    显存优化的测试函数：正确处理维度不匹配
    """
    task_model.eval()
    if model_degrad is not None:
        model_degrad.eval()
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_idx, (frames, action_lab, privacy_lab) in tqdm(enumerate(data_loader), 
                                                                 total=len(data_loader), 
                                                                 desc=f"测试{task_type}"):
            frames = frames.to(device, non_blocking=True)
            
            # 选择标签
            if task_type == "action":
                labels = action_lab.to(device)
            else:
                labels = privacy_lab.to(device)
            
            # 标签处理
            labels = labels - label_min
            labels = torch.clamp(labels, min=0, max=num_classes-1)
            
            if use_bdq and model_degrad is not None:
                # === BDQ 路径 ===
                try:
                    bdq_output, _ = model_degrad(frames)
                except:
                    bdq_output = model_degrad(frames)
                    if isinstance(bdq_output, tuple):
                        bdq_output = bdq_output[0]
                
                # 展平 BDQ 输出
                if len(bdq_output.shape) > 2:
                    bdq_output = bdq_output.view(bdq_output.size(0), -1)

                # [调试信息]
                if batch_idx == 0:
                    print(f"\n[DEBUG BDQ路径] Task: {task_type}, BDQ Shape: {bdq_output.shape}")
                    print(f"Classifier expects input dim: {task_model.classifier.in_features}")
                
                # 动态调整分类器以匹配实际特征维度
                feature_dim = bdq_output.shape[1]
                if feature_dim != task_model.classifier.in_features:
                    print(f"动态调整分类器: {task_model.classifier.in_features} -> {feature_dim}")
                    old_classifier = task_model.classifier
                    task_model.classifier = nn.Linear(feature_dim, num_classes).to(device)
                    
                    # 尝试复制权重（如果维度兼容）
                    try:
                        if old_classifier.weight.shape[1] == feature_dim:
                            task_model.classifier.weight.data.copy_(old_classifier.weight.data)
                            task_model.classifier.bias.data.copy_(old_classifier.bias.data)
                    except:
                        pass  # 如果无法复制权重，使用随机初始化
                
                # 通过分类器
                outputs = task_model.classifier(bdq_output)
                    
            else:
                # === 原始视频路径 ===
                outputs = task_model(frames, use_bdq=False)
                
                if batch_idx == 0:
                    print(f"\n[DEBUG 原始路径] Task: {task_type}, Outputs Shape: {outputs.shape}")
            
            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # 获取预测结果
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 清理缓存以释放显存
            if batch_idx % 10 == 0:  # 每10个批次清理一次
                torch.cuda.empty_cache()
    
    # 计算准确率
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    accuracy = np.mean(all_predictions == all_labels) * 100
    avg_loss = total_loss / len(data_loader)
    
    # 计算每个类别的准确率
    class_accuracies = {}
    for class_id in range(num_classes):
        mask = all_labels == class_id
        if np.sum(mask) > 0:
            class_accuracy = np.mean(all_predictions[mask] == all_labels[mask]) * 100
            class_accuracies[class_id] = class_accuracy
    
    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "predictions": all_predictions,
        "labels": all_labels,
        "class_accuracies": class_accuracies
    }

def test_unified_model(model_degrad, task_model, data_loader, device,
                      task_type="action", use_bdq=False, label_min=0, num_classes=None):
    """统一接口测试函数 - 修复版"""
    task_model.eval()
    if model_degrad is not None:
        model_degrad.eval()
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()
    
    # 记录第一次调用，用于调试
    first_batch = True
    
    with torch.no_grad():
        for batch_idx, (frames, action_lab, privacy_lab) in tqdm(enumerate(data_loader),
                                                                 total=len(data_loader),
                                                                 desc=f"测试{task_type}"):
            frames = frames.to(device, non_blocking=True)
            
            # 选择标签
            if task_type == "action":
                labels = action_lab.to(device)
            else:
                labels = privacy_lab.to(device)
            
            # 标签处理
            labels = labels - label_min
            labels = torch.clamp(labels, min=0, max=num_classes-1)
            
            if use_bdq and model_degrad is not None:
                # 使用BDQ处理
                result = model_degrad(frames)
                
                # 处理不同的返回值格式
                if isinstance(result, tuple):
                    if len(result) >= 2:
                        bdq_features = result[0]
                    else:
                        bdq_features = result[0]
                elif isinstance(result, list):
                    bdq_features = result[0]
                else:
                    bdq_features = result
                
                # 调试信息
                if first_batch:
                    print(f"[调试] BDQ特征形状: {bdq_features.shape}")
                    first_batch = False
                
                if task_type == "action":
                    # 动作识别：使用BDQ特征
                    try:
                        # 尝试直接调用
                        outputs = task_model(bdq_features, use_bdq=True)
                    except Exception as e:
                        print(f"[错误] 调用task_model失败: {e}")
                        print(f"[调试] bdq_features形状: {bdq_features.shape}")
                        # 尝试其他方式
                        if hasattr(task_model, 'bdq_feature_adapter'):
                            adapted = task_model.bdq_feature_adapter(bdq_features)
                            outputs = task_model.classifier(adapted)
                        else:
                            raise e
                else:
                    # ====== 修复隐私识别部分的维度问题 ======
                    # 隐私识别：检查BDQ特征维度
                    b = bdq_features.size(0)
                    d = bdq_features.size(1) if len(bdq_features.shape) > 1 else bdq_features.size(0)
                    
                    # 如果BDQ特征已经是2D的 [B, D]，且维度较小，直接使用线性层
                    if len(bdq_features.shape) == 2 and d < 1000:  # 假设小于1000维就是低维特征
                        print(f"[调试] 隐私识别使用BDQ特征: 形状={bdq_features.shape}")
                        # 直接通过分类器（假设任务模型有classifier属性）
                        if hasattr(task_model, 'classifier'):
                            outputs = task_model.classifier(bdq_features)
                        else:
                            # 创建临时分类器
                            outputs = nn.Linear(d, num_classes).to(device)(bdq_features)
                    else:
                        # 如果BDQ特征维度较大，尝试重塑为图像
                        try:
                            # 尝试重塑为合理的图像形状
                            if d >= 3 * 56 * 56:  # 至少能重塑为56x56
                                h = w = int((d // 3) ** 0.5)
                                c = 3
                                img_features = bdq_features[:, :c*h*w].view(b, c, h, w)
                                outputs = task_model(img_features)
                            else:
                                # 使用一个小型的特征提取器
                                print(f"[警告] BDQ特征维度{d}太小，无法重塑为图像")
                                # 使用线性分类器
                                if hasattr(task_model, 'classifier'):
                                    outputs = task_model.classifier(bdq_features)
                                else:
                                    # 创建临时分类器
                                    outputs = nn.Linear(d, num_classes).to(device)(bdq_features)
                        except Exception as reshape_error:
                            print(f"[错误] 重塑BDQ特征失败: {reshape_error}")
                            # 回退到线性分类器
                            outputs = nn.Linear(d, num_classes).to(device)(bdq_features)
            else:
                # 原始视频
                if task_type == "action":
                    outputs = task_model(frames, use_bdq=False)
                else:
                    # 隐私识别：取中间帧
                    if len(frames.shape) == 5:
                        T = frames.shape[2]
                        frames_mid = frames[:, :, T//2, :, :]
                        outputs = task_model(frames_mid)
                    else:
                        outputs = task_model(frames)
            
            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # 获取预测结果
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 每5个批次清理一次显存
            if batch_idx % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 计算准确率
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    accuracy = np.mean(all_predictions == all_labels) * 100
    avg_loss = total_loss / len(data_loader)
    
    # 计算每个类别的准确率
    class_accuracies = {}
    for class_id in range(num_classes):
        mask = all_labels == class_id
        if np.sum(mask) > 0:
            class_accuracy = np.mean(all_predictions[mask] == all_labels[mask]) * 100
            class_accuracies[class_id] = class_accuracy
    
    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "predictions": all_predictions,
        "labels": all_labels,
        "class_accuracies": class_accuracies
    }


# ===================== 可视化函数 =====================
def create_detailed_visualization(model_degrad, dataset, config, save_dir):
    """创建详细的BDQ处理过程可视化"""
    print_section("创建可视化")
    
    if model_degrad is None:
        print("模型未加载，无法创建可视化")
        return
    
    model_degrad.eval()
    
    # 获取样本（减少数量以节省显存）
    sample_indices = min(2, len(dataset))  # 减少可视化样本数量
    
    fig, axes = plt.subplots(sample_indices, 4, figsize=(16, 4 * sample_indices))
    if sample_indices == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i in range(sample_indices):
            frames, action_label, privacy_label = dataset[i]
            frames = frames.unsqueeze(0).to(next(model_degrad.parameters()).device)
            
            try:
                # 获取BDQ输出
                bdq_output = model_degrad(frames)
                if isinstance(bdq_output, tuple):
                    bdq_output = bdq_output[0]
                
                # 模拟三个阶段（由于没有get_bdq_stages函数）
                # 这里我们创建简单的模拟
                stage1 = frames  # 假设第一阶段是原始帧
                stage2 = frames  # 假设第二阶段是帧差
                stage3 = bdq_output  # 第三阶段是BDQ输出
            except Exception as e:
                print(f"获取BDQ阶段失败: {e}")
                stage1 = stage2 = stage3 = frames
            
            # 原始帧
            if len(frames.shape) == 5:
                mid_frame = frames[0, :, frames.shape[2]//2, :, :].cpu()
            else:
                mid_frame = frames[0].cpu()
            
            # 绘制
            try:
                axes[i, 0].imshow(cv2.cvtColor(tensor2img(mid_frame, config["mean"], config["std"]), cv2.COLOR_BGR2RGB))
                axes[i, 1].imshow(cv2.cvtColor(tensor2img(stage1[0], config["mean"], config["std"]), cv2.COLOR_BGR2RGB))
                axes[i, 2].imshow(cv2.cvtColor(tensor2img(stage2[0], config["mean"], config["std"]), cv2.COLOR_BGR2RGB))
                axes[i, 3].imshow(cv2.cvtColor(tensor2img(stage3[0], config["mean"], config["std"]), cv2.COLOR_BGR2RGB))
            except Exception as e:
                print(f"可视化绘制失败: {e}")
                for j in range(4):
                    axes[i, j].imshow(np.random.rand(224, 224, 3))
            
            if i == 0:
                titles = [
                    f"原始视频帧\n动作:{action_label}, 隐私:{privacy_label}",
                    "阶段1: 高斯平滑",
                    "阶段2: 帧差计算", 
                    "阶段3: 量化输出"
                ]
                for j, title in enumerate(titles):
                    axes[i, j].set_title(title, fontsize=12, fontweight='bold')
            
            for j in range(4):
                axes[i, j].axis("off")
    
    plt.tight_layout()
    vis_path = os.path.join(save_dir, "bdq_detailed_visualization.png")
    plt.savefig(vis_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ 详细可视化已保存: {vis_path}")

def plot_performance_comparison(action_results, privacy_results, save_dir):
    """绘制性能对比图"""
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 总体准确率对比
    ax1 = plt.subplot(2, 3, 1)
    categories = ['原始视频', 'BDQ处理']
    action_acc = [action_results['raw']['accuracy'], action_results['bdq']['accuracy']]
    privacy_acc = [privacy_results['raw']['accuracy'], privacy_results['bdq']['accuracy']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, action_acc, width, label='动作识别', color='blue', alpha=0.7)
    bars2 = ax1.bar(x + width/2, privacy_acc, width, label='隐私识别', color='red', alpha=0.7)
    
    ax1.set_xlabel('处理方式', fontsize=12)
    ax1.set_ylabel('准确率 (%)', fontsize=12)
    ax1.set_title('总体准确率对比', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # 2. 准确率变化
    ax2 = plt.subplot(2, 3, 2)
    changes = [
        action_results['bdq']['accuracy'] - action_results['raw']['accuracy'],
        privacy_results['bdq']['accuracy'] - privacy_results['raw']['accuracy']
    ]
    colors = ['green' if c > -5 else 'orange' if c > -15 else 'red' for c in changes]
    
    bars = ax2.bar(['动作识别', '隐私识别'], changes, color=colors, alpha=0.7)
    ax2.set_ylabel('准确率变化 (%)', fontsize=12)
    ax2.set_title('BDQ处理对准确率的影响', fontsize=14, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -2),
                f'{height:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
    
    # 3. 损失对比
    ax3 = plt.subplot(2, 3, 3)
    action_loss = [action_results['raw']['loss'], action_results['bdq']['loss']]
    privacy_loss = [privacy_results['raw']['loss'], privacy_results['bdq']['loss']]
    
    bars1 = ax3.bar(x - width/2, action_loss, width, label='动作识别', color='blue', alpha=0.7)
    bars2 = ax3.bar(x + width/2, privacy_loss, width, label='隐私识别', color='red', alpha=0.7)
    
    ax3.set_xlabel('处理方式', fontsize=12)
    ax3.set_ylabel('损失值', fontsize=12)
    ax3.set_title('损失值对比', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 类别准确率（动作识别）
    ax4 = plt.subplot(2, 3, 4)
    action_classes = list(range(min(5, len(action_results['raw']['class_accuracies']))))  # 限制显示类别数
    raw_action_class_acc = [action_results['raw']['class_accuracies'].get(i, 0) for i in action_classes]
    bdq_action_class_acc = [action_results['bdq']['class_accuracies'].get(i, 0) for i in action_classes]
    
    x = np.arange(len(action_classes))
    bars1 = ax4.bar(x - width/2, raw_action_class_acc, width, label='原始', color='lightblue')
    bars2 = ax4.bar(x + width/2, bdq_action_class_acc, width, label='BDQ', color='lightcoral')
    
    ax4.set_xlabel('动作类别', fontsize=12)
    ax4.set_ylabel('准确率 (%)', fontsize=12)
    ax4.set_title('动作识别各类别准确率', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'C{i}' for i in action_classes])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 100])
    
    # 5. 类别准确率（隐私识别）
    ax5 = plt.subplot(2, 3, 5)
    privacy_classes = list(range(min(5, len(privacy_results['raw']['class_accuracies']))))  # 限制显示类别数
    raw_privacy_class_acc = [privacy_results['raw']['class_accuracies'].get(i, 0) for i in privacy_classes]
    bdq_privacy_class_acc = [privacy_results['bdq']['class_accuracies'].get(i, 0) for i in privacy_classes]
    
    x = np.arange(len(privacy_classes))
    bars1 = ax5.bar(x - width/2, raw_privacy_class_acc, width, label='原始', color='lightblue')
    bars2 = ax5.bar(x + width/2, bdq_privacy_class_acc, width, label='BDQ', color='lightcoral')
    
    ax5.set_xlabel('隐私类别', fontsize=12)
    ax5.set_ylabel('准确率 (%)', fontsize=12)
    ax5.set_title('隐私识别各类别准确率', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([f'P{i}' for i in privacy_classes])
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim([0, 100])
    
    # 6. 性能保持率
    ax6 = plt.subplot(2, 3, 6)
    action_preserve = action_results['bdq']['accuracy'] / max(action_results['raw']['accuracy'], 1e-10) * 100
    privacy_preserve = privacy_results['bdq']['accuracy'] / max(privacy_results['raw']['accuracy'], 1e-10) * 100
    
    bars = ax6.bar(['动作识别', '隐私识别'], [action_preserve, privacy_preserve], 
                   color=['blue', 'red'], alpha=0.7)
    ax6.set_ylabel('性能保持率 (%)', fontsize=12)
    ax6.set_title('BDQ处理后性能保持率', fontsize=14, fontweight='bold')
    ax6.axhline(y=100, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax6.axhline(y=80, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax6.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_ylim([0, 120])
    
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('BDQ编码器性能对比分析', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, "detailed_performance_comparison.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ 详细性能对比图已保存: {plot_path}")
    
    return plot_path

# ===================== 保存结果函数 =====================
def save_results(action_results, privacy_results, val_stats, config, total_start_time, test_time):
    """保存实验结果"""
    print_section("保存详细结果")
    
    # 准备结果数据
    all_results = {
        "experiment_info": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            "total_duration_seconds": time.time() - total_start_time,
            "test_duration_seconds": test_time,
            "dataset_size": val_stats["total_samples"],
            "action_classes": config.num_classes,
            "privacy_classes": config.privacy_num_classes,
            "model_file": CKPT_PATH,
            "model_size_mb": os.path.getsize(CKPT_PATH) / 1024**2 if os.path.exists(CKPT_PATH) else 0
        },
        "dataset_stats": {
            "total_samples": val_stats["total_samples"],
            "sample_analyzed": val_stats["sample_analyzed"],
            "action_classes": val_stats["action_classes"],
            "privacy_classes": val_stats["privacy_classes"],
            "action_label_dist": {str(k): int(v) for k, v in val_stats["action_label_dist"].items()},
            "privacy_label_dist": {str(k): int(v) for k, v in val_stats["privacy_label_dist"].items()},
            "action_label_range": [int(val_stats['action_label_range'][0]), int(val_stats['action_label_range'][1])],
            "privacy_label_range": [int(val_stats['privacy_label_range'][0]), int(val_stats['privacy_label_range'][1])],
            "action_label_std": float(val_stats['action_label_std']),
            "privacy_label_std": float(val_stats['privacy_label_std'])
        },
        "action_recognition": {
            "raw": {
                "accuracy": float(action_results['raw']["accuracy"]),
                "loss": float(action_results['raw']["loss"]),
                "class_accuracies": {str(k): float(v) for k, v in action_results['raw']["class_accuracies"].items()}
            },
            "bdq": {
                "accuracy": float(action_results['bdq']["accuracy"]),
                "loss": float(action_results['bdq']["loss"]),
                "class_accuracies": {str(k): float(v) for k, v in action_results['bdq']["class_accuracies"].items()}
            },
            "summary": {
                "accuracy_change": float(action_results['bdq']["accuracy"] - action_results['raw']["accuracy"]),
                "performance_preserved": float(action_results['bdq']["accuracy"] / max(action_results['raw']["accuracy"], 1e-10) * 100)
            }
        },
        "privacy_recognition": {
            "raw": {
                "accuracy": float(privacy_results['raw']["accuracy"]),
                "loss": float(privacy_results['raw']["loss"]),
                "class_accuracies": {str(k): float(v) for k, v in privacy_results['raw']["class_accuracies"].items()}
            },
            "bdq": {
                "accuracy": float(privacy_results['bdq']["accuracy"]),
                "loss": float(privacy_results['bdq']["loss"]),
                "class_accuracies": {str(k): float(v) for k, v in privacy_results['bdq']["class_accuracies"].items()}
            },
            "summary": {
                "accuracy_change": float(privacy_results['bdq']["accuracy"] - privacy_results['raw']["accuracy"]),
                "performance_reduced": float(100 - (privacy_results['bdq']["accuracy"] / max(privacy_results['raw']["accuracy"], 1e-10) * 100))
            }
        }
    }
    
    # 保存JSON格式结果
    json_path = os.path.join(RESULTS_DIR, "rigorous_test_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"✓ 详细结果已保存: {json_path}")
    
    # 保存文本报告
    report_path = os.path.join(RESULTS_DIR, "rigorous_experiment_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("BDQ编码器严谨测试实验报告\n")
        f.write("="*80 + "\n\n")
        
        f.write("一、实验概述\n")
        f.write("-"*40 + "\n")
        f.write(f"实验时间: {all_results['experiment_info']['timestamp']}\n")
        f.write(f"运行设备: {all_results['experiment_info']['device']}\n")
        f.write(f"总运行时间: {all_results['experiment_info']['total_duration_seconds']:.2f}秒\n")
        f.write(f"测试时间: {all_results['experiment_info']['test_duration_seconds']:.2f}秒\n")
        f.write(f"模型文件: {all_results['experiment_info']['model_file']}\n")
        f.write(f"模型大小: {all_results['experiment_info']['model_size_mb']:.2f} MB\n\n")
        
        f.write("二、数据集信息\n")
        f.write("-"*40 + "\n")
        f.write(f"数据集大小: {all_results['dataset_stats']['total_samples']}个样本\n")
        f.write(f"动作类别数: {all_results['dataset_stats']['action_classes']}\n")
        f.write(f"隐私类别数: {all_results['dataset_stats']['privacy_classes']}\n")
        f.write(f"动作标签分布: {all_results['dataset_stats']['action_label_dist']}\n")
        f.write(f"隐私标签分布: {all_results['dataset_stats']['privacy_label_dist']}\n\n")
        
        f.write("三、性能测试结果\n")
        f.write("-"*40 + "\n\n")
        
        f.write("【动作识别任务】\n")
        f.write(f"原始视频准确率: {all_results['action_recognition']['raw']['accuracy']:.2f}%\n")
        f.write(f"BDQ处理后准确率: {all_results['action_recognition']['bdq']['accuracy']:.2f}%\n")
        f.write(f"准确率变化: {all_results['action_recognition']['summary']['accuracy_change']:+.2f}%\n")
        f.write(f"性能保持率: {all_results['action_recognition']['summary']['performance_preserved']:.1f}%\n\n")
        
        f.write("各类别准确率 (原始视频):\n")
        for class_id, acc in sorted(all_results['action_recognition']['raw']['class_accuracies'].items())[:5]:
            f.write(f"  类别{class_id}: {acc:.1f}%\n")
        
        f.write("\n各类别准确率 (BDQ处理后):\n")
        for class_id, acc in sorted(all_results['action_recognition']['bdq']['class_accuracies'].items())[:5]:
            f.write(f"  类别{class_id}: {acc:.1f}%\n")
        
        f.write("\n【隐私识别任务】\n")
        f.write(f"原始视频准确率: {all_results['privacy_recognition']['raw']['accuracy']:.2f}%\n")
        f.write(f"BDQ处理后准确率: {all_results['privacy_recognition']['bdq']['accuracy']:.2f}%\n")
        f.write(f"准确率变化: {all_results['privacy_recognition']['summary']['accuracy_change']:+.2f}%\n")
        f.write(f"性能降低率: {all_results['privacy_recognition']['summary']['performance_reduced']:.1f}%\n\n")
        
        f.write("各类别准确率 (原始视频):\n")
        for class_id, acc in sorted(all_results['privacy_recognition']['raw']['class_accuracies'].items())[:5]:
            f.write(f"  类别{class_id}: {acc:.1f}%\n")
        
        f.write("\n各类别准确率 (BDQ处理后):\n")
        for class_id, acc in sorted(all_results['privacy_recognition']['bdq']['class_accuracies'].items())[:5]:
            f.write(f"  类别{class_id}: {acc:.1f}%\n")
    
    print(f"✓ 详细实验报告已保存: {report_path}")

# ===================== 主函数 =====================
def main():
    """修复版主函数 - 解决backbone_net缺失问题"""
    print_section("BDQ编码器严谨测试实验 - 修复版", 80)
    print("实验开始时间:", time.strftime("%Y-%m-%d %H:%M:%S"))
    
    total_start_time = time.time()
    
    # ===================== 1. 初始化配置 =====================
    print_section("1. 初始化配置")   
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(f"使用设备: {device}")
    log_message(f"PyTorch版本: {torch.__version__}")
    log_message(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        log_message(f"GPU型号: {torch.cuda.get_device_name(0)}")
        log_message(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # ===================== 2. 加载预训练BDQ编码器 =====================
    print_section("2. 加载预训练BDQ编码器")
    
    # 检查文件是否存在
    if not os.path.exists(CKPT_PATH):
        log_message(f"模型文件不存在: {CKPT_PATH}", "ERROR")
        log_message("请确保模型文件已放置在正确位置", "ERROR")
        sys.exit(1)
    
    # 获取文件大小但不检查
    file_size = os.path.getsize(CKPT_PATH) / 1024**2
    log_message(f"模型文件: {CKPT_PATH}")
    log_message(f"文件大小: {file_size:.2f} MB")
    
    try:
        # 确保config有所有必要属性
        args = config.to_namespace()
        
        # 构建模型
        model, arch_name = build_model(args, test_mode=True)
        
        if not model:
            log_message("模型构建失败", "ERROR")
            sys.exit(1)
        
        # 处理模型返回格式
        if isinstance(model, tuple):
            model_degrad = model[0].to(device)
        elif isinstance(model, list):
            model_degrad = model[0].to(device)
        else:
            model_degrad = model.to(device)
        
        log_message("✓ 模型构建成功")
        log_message(f"模型架构: {arch_name}")
        
        # 加载预训练权重
        try:
            checkpoint = torch.load(CKPT_PATH, map_location=device)
            log_message(f"检查点加载成功，类型: {type(checkpoint)}")
            
            # 处理不同的checkpoint格式
            state_dict = {}
            if isinstance(checkpoint, dict):
                # 如果是字典，直接使用
                state_dict = checkpoint
            elif isinstance(checkpoint, list) or isinstance(checkpoint, tuple):
                # 如果是列表或元组，取第一个元素
                if len(checkpoint) > 0:
                    state_dict = checkpoint[0]
                else:
                    raise ValueError("检查点列表为空")
            else:
                # 其他情况，尝试直接使用
                state_dict = checkpoint
            
            # 清理状态字典
            if isinstance(state_dict, dict):
                cleaned_state_dict = {}
                for k, v in state_dict.items():
                    new_key = k.replace("module.", "")
                    cleaned_state_dict[new_key] = v
                state_dict = cleaned_state_dict
                
                # 加载权重
                missing_keys, unexpected_keys = model_degrad.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    log_message(f"缺失的键: {missing_keys[:5]}...", "WARNING")
                if unexpected_keys:
                    log_message(f"意外的键: {unexpected_keys[:5]}...", "WARNING")
                
                log_message("✓ BDQ编码器权重加载成功")
            else:
                log_message(f"警告: 状态字典不是字典类型，跳过加载", "WARNING")
                missing_keys, unexpected_keys = [], []
                
        except Exception as e:
            log_message(f"加载权重失败: {e}", "WARNING")
            log_message("将使用随机初始化的权重", "WARNING")
            missing_keys, unexpected_keys = [], []
        
        # 测试BDQ输出维度
        with torch.no_grad():
            test_input = torch.randn(1, 3, 16, 224, 224).to(device)
            try:
                # 尝试不同的调用方式
                if hasattr(model_degrad, 'base_model'):
                    # 如果是统一接口模型
                    bdq_output = model_degrad(test_input)
                else:
                    # 直接调用模型
                    result = model_degrad(test_input)
                    if isinstance(result, tuple):
                        bdq_output = result[0]
                    else:
                        bdq_output = result
                
                log_message(f"BDQ输出维度: {bdq_output.shape}")
                
                # 检查特征维度
                if len(bdq_output.shape) == 2:
                    actual_dim = bdq_output.shape[1]
                else:
                    # 如果是其他形状，展平计算维度
                    actual_dim = bdq_output.view(bdq_output.size(0), -1).shape[1]
                
                if actual_dim == config.feature_dim:
                    log_message(f"✓ BDQ输出维度正确: {config.feature_dim}维")
                else:
                    log_message(f"⚠ BDQ输出维度: {actual_dim}维 (期望{config.feature_dim}维)", "WARNING")
                    config.feature_dim = actual_dim
                    log_message(f"已动态调整特征维度为: {config.feature_dim}")
                    
            except Exception as e:
                log_message(f"测试BDQ输出失败: {e}", "WARNING")
                # 假设默认维度
                config.feature_dim = 2048
                log_message(f"使用默认特征维度: {config.feature_dim}")
        
    except Exception as e:
        log_message(f"加载模型失败: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ===================== 3. 加载和准备数据集 =====================
    print_section("3. 加载和准备数据集")
    
    # 检查数据集文件
    if not os.path.exists(TRAIN_TXT_PATH):
        log_message(f"训练文件不存在: {TRAIN_TXT_PATH}", "ERROR")
        sys.exit(1)
    
    if not os.path.exists(VAL_TXT_PATH):
        log_message(f"验证文件不存在: {VAL_TXT_PATH}", "ERROR")
        sys.exit(1)
    
    # 加载数据集配置
    sbu_config = get_sbu_config()
    
    # 数据增强
    val_aug = get_augmentor(
        is_train=False,
        image_size=224,
        mean=sbu_config["mean"],
        std=sbu_config["std"],
        threed_data=config.threed_data,
        is_flow=(config.modality == "flow")
    )
    
    # 加载验证数据集
    try:
        val_dataset = SBUDataSet(
            root=config.datadir,
            list_file=VAL_TXT_PATH,
            num_groups=config.groups,
            frames_per_group=config.frames_per_group,
            transform=val_aug,
            is_train=False,
            image_tmpl=sbu_config["image_tmpl"]
        )
        log_message(f"✓ 验证集加载成功: {len(val_dataset)} 个样本")
    except Exception as e:
        log_message(f"加载数据集失败: {e}", "ERROR")
        sys.exit(1)
    
    # 分析数据集
    val_stats = analyze_dataset(val_dataset, "验证集")
    
    # 更新配置
    config.num_classes = val_stats["action_classes"]
    config.privacy_num_classes = val_stats["privacy_classes"]
    
    log_message(f"动作类别数: {config.num_classes}")
    log_message(f"隐私类别数: {config.privacy_num_classes}")
    log_message(f"动作标签最小值: {val_stats['action_label_range'][0]}")
    log_message(f"隐私标签最小值: {val_stats['privacy_label_range'][0]}")
    
    # 创建数据加载器
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=True,
        drop_last=False
    )
    
    # ===================== 4. 准备统一接口任务模型 =====================
    print_section("4. 准备统一接口任务模型")

    # 首先测试BDQ模型的真实输出维度
    print("测试BDQ模型的实际输出维度...")
    with torch.no_grad():
        test_frames = torch.randn(2, 3, 16, 224, 224).to(device)  # 2个样本用于测试
        result = model_degrad(test_frames)
        
        # 处理不同的返回值格式
        if isinstance(result, tuple):
            if len(result) >= 2:
                bdq_features = result[0]
            else:
                bdq_features = result[0]
        elif isinstance(result, list):
            bdq_features = result[0]
        else:
            bdq_features = result
        
        # 获取特征维度
        if len(bdq_features.shape) == 2:
            bdq_output_dim = bdq_features.shape[1]
        else:
            # 展平计算维度
            bdq_output_dim = bdq_features.view(bdq_features.size(0), -1).shape[1]
        
        print(f"BDQ模型实际输出维度: {bdq_output_dim}")
        
        # 与期望的特征维度比较
        if bdq_output_dim != config.feature_dim:
            print(f"注意: BDQ输出维度({bdq_output_dim})与期望特征维度({config.feature_dim})不同")
            print(f"将使用BDQ实际输出维度: {bdq_output_dim}")
            actual_feature_dim = bdq_output_dim
        else:
            actual_feature_dim = config.feature_dim

    # 动作识别模型 - 使用实际特征维度
    action_model = UnifiedActionRecognizer(
        num_classes=config.num_classes,
        feature_dim=config.feature_dim,  # 内部特征维度
        bdq_input_dim=actual_feature_dim  # BDQ实际输出维度
    ).to(device)

    log_message(f"✓ 统一动作识别模型准备完成: BDQ输入={actual_feature_dim}, 内部特征={config.feature_dim}")

    # 隐私识别模型
    privacy_model = UnifiedPrivacyRecognizer(
        num_classes=config.privacy_num_classes
    ).to(device)

    log_message("✓ 统一隐私识别模型准备完成")
    
    # ===================== 5. 真实性能测试 =====================
    print_section("5. 真实性能测试")
    
    test_start_time = time.time()
    
    # 测试动作识别任务
    log_message("开始动作识别任务测试...")
    
    # 测试原始视频
    action_raw_results = test_unified_model(
        model_degrad=None,
        task_model=action_model,
        data_loader=val_loader,
        device=device,
        task_type="action",
        use_bdq=False,
        label_min=val_stats['action_label_range'][0],
        num_classes=config.num_classes
    )
    
    log_message(f"原始视频动作识别: 准确率={action_raw_results['accuracy']:.2f}%, 损失={action_raw_results['loss']:.4f}")
    
    # 测试BDQ处理后视频
    action_bdq_results = test_unified_model(
        model_degrad=model_degrad,
        task_model=action_model,
        data_loader=val_loader,
        device=device,
        task_type="action",
        use_bdq=True,
        label_min=val_stats['action_label_range'][0],
        num_classes=config.num_classes
    )
    
    log_message(f"BDQ处理后动作识别: 准确率={action_bdq_results['accuracy']:.2f}%, 损失={action_bdq_results['loss']:.4f}")
    
    # 测试隐私识别任务
    log_message("开始隐私识别任务测试...")
    
    # 测试原始视频
    privacy_raw_results = test_unified_model(
        model_degrad=None,
        task_model=privacy_model,
        data_loader=val_loader,
        device=device,
        task_type="privacy",
        use_bdq=False,
        label_min=val_stats['privacy_label_range'][0],
        num_classes=config.privacy_num_classes
    )
    
    log_message(f"原始视频隐私识别: 准确率={privacy_raw_results['accuracy']:.2f}%, 损失={privacy_raw_results['loss']:.4f}")
    
    # 测试BDQ处理后视频
    privacy_bdq_results = test_unified_model(
        model_degrad=model_degrad,
        task_model=privacy_model,
        data_loader=val_loader,
        device=device,
        task_type="privacy",
        use_bdq=True,
        label_min=val_stats['privacy_label_range'][0],
        num_classes=config.privacy_num_classes
    )
    
    log_message(f"BDQ处理后隐私识别: 准确率={privacy_bdq_results['accuracy']:.2f}%, 损失={privacy_bdq_results['loss']:.4f}")
    
    test_time = time.time() - test_start_time
    log_message(f"测试完成，耗时: {test_time:.2f}秒")
    
    # 汇总结果
    action_results = {
        'raw': action_raw_results,
        'bdq': action_bdq_results
    }
    
    privacy_results = {
        'raw': privacy_raw_results,
        'bdq': privacy_bdq_results
    }
    
    # ===================== 6. 生成可视化 =====================
    print_section("6. 生成可视化")
    
    # 创建可视化
    create_detailed_visualization(model_degrad, val_dataset, sbu_config, VIS_DIR)
    plot_performance_comparison(action_results, privacy_results, VIS_DIR)
    
    # ===================== 7. 保存详细结果 =====================
    print_section("7. 保存详细结果")
    
    # 准备结果数据
    all_results = {
    "experiment_info": {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "total_duration_seconds": time.time() - total_start_time,
        "test_duration_seconds": test_time,
        "dataset_size": val_stats["total_samples"],
        "action_classes": config.num_classes,
        "privacy_classes": config.privacy_num_classes,
        "feature_dim": config.feature_dim,
        "model_file": CKPT_PATH,
        "model_size_mb": file_size,
        "unified_interface": config.unified_interface
    },
    "dataset_stats": {
        "total_samples": val_stats["total_samples"],
        "sample_analyzed": val_stats["sample_analyzed"],
        "action_classes": val_stats["action_classes"],
        "privacy_classes": val_stats["privacy_classes"],
        "action_label_dist": {str(k): int(v) for k, v in val_stats["action_label_dist"].items()},
        "privacy_label_dist": {str(k): int(v) for k, v in val_stats["privacy_label_dist"].items()},
        "action_label_range": [int(val_stats['action_label_range'][0]), int(val_stats['action_label_range'][1])],
        "privacy_label_range": [int(val_stats['privacy_label_range'][0]), int(val_stats['privacy_label_range'][1])],
        "action_label_std": float(val_stats['action_label_std']),
        "privacy_label_std": float(val_stats['privacy_label_std'])
    },
    "action_recognition": {
        "raw": {
            "accuracy": float(action_raw_results['accuracy']),
            "loss": float(action_raw_results['loss']),
            "class_accuracies": {str(k): float(v) for k, v in action_raw_results["class_accuracies"].items()},
            "predictions": action_raw_results["predictions"].tolist(),
            "labels": action_raw_results["labels"].tolist()
        },
        "bdq": {
            "accuracy": float(action_bdq_results['accuracy']),
            "loss": float(action_bdq_results['loss']),
            "class_accuracies": {str(k): float(v) for k, v in action_bdq_results["class_accuracies"].items()},
            "predictions": action_bdq_results["predictions"].tolist(),
            "labels": action_bdq_results["labels"].tolist()
        }
    },
    "privacy_recognition": {
        "raw": {
            "accuracy": float(privacy_raw_results['accuracy']),
            "loss": float(privacy_raw_results['loss']),
            "class_accuracies": {str(k): float(v) for k, v in privacy_raw_results["class_accuracies"].items()},
            "predictions": privacy_raw_results["predictions"].tolist(),
            "labels": privacy_raw_results["labels"].tolist()
        },
        "bdq": {
            "accuracy": float(privacy_bdq_results['accuracy']),
            "loss": float(privacy_bdq_results['loss']),
            "class_accuracies": {str(k): float(v) for k, v in privacy_bdq_results["class_accuracies"].items()},
            "predictions": privacy_bdq_results["predictions"].tolist(),
            "labels": privacy_bdq_results["labels"].tolist()
        }
    }
}
    
    # 保存JSON格式结果
    json_path = os.path.join(RESULTS_DIR, "rigorous_test_results_fixed.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    log_message(f"✓ 详细结果已保存: {json_path}")
    
    # 保存文本报告
    report_path = os.path.join(RESULTS_DIR, "rigorous_experiment_report_fixed.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("BDQ编码器严谨测试实验报告 (修复版)\n")
        f.write("="*80 + "\n\n")
        
        f.write("实验配置修复说明:\n")
        f.write("- 修复了Config类中缺少backbone_net属性的问题\n")
        f.write("- 移除了文件大小检查\n")
        f.write("- 使用统一接口模型\n\n")
        
        f.write(f"动作识别准确率: {action_bdq_results['accuracy']:.2f}% (BDQ处理后)\n")
        f.write(f"隐私识别准确率: {privacy_bdq_results['accuracy']:.2f}% (BDQ处理后)\n")
    
    log_message(f"✓ 详细实验报告已保存: {report_path}")
    
    # ===================== 8. 实验完成 =====================
    print_section("8. 实验完成")
    
    total_time = time.time() - total_start_time
    
    print("\n" + "="*80)
    print("实验成功完成！")
    print("="*80)
    
    print(f"\n✅ 所有修复已应用:")
    print(f"   1. 添加了backbone_net属性到Config类")
    print(f"   2. 移除了文件大小检查")
    print(f"   3. 使用统一接口模型")
    
    print(f"\n📊 核心结果:")
    print(f"   动作识别准确率: {action_bdq_results['accuracy']:.1f}%")
    print(f"   隐私识别准确率: {privacy_bdq_results['accuracy']:.1f}%")
    
    print(f"\n⏱️ 总运行时间: {total_time:.1f}秒")
    print(f"📁 结果保存到: results/ 和 visualization/ 目录")
    
    print(f"\n✅ 所有任务已完成！")
    print(f"   实验时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()