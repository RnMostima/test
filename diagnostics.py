# diagnostics.py
import torch
import numpy as np
import os
import sys

def analyze_checkpoint(ckpt_path):
    """分析预训练检查点"""
    print("="*80)
    print("检查点分析报告")
    print("="*80)
    
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        print(f"加载检查点失败: {e}")
        return
    
    print(f"\n🔍 检查点键值:")
    for key in checkpoint.keys():
        if hasattr(checkpoint[key], 'shape'):
            print(f"  {key}: {checkpoint[key].shape}")
        else:
            print(f"  {key}: {type(checkpoint[key])}")
    
    # 分析可能包含8192维的层
    print(f"\n🔍 寻找8192维相关的层:")
    found_8192 = False
    for key, value in checkpoint.items():
        if hasattr(value, 'shape'):
            if 8192 in value.shape:
                print(f"  {key}: {value.shape} (包含8192)")
                found_8192 = True
    
    if not found_8192:
        print("  未找到8192维度的层")
    
    print(f"\n🔍 寻找2048维相关的层:")
    found_2048 = False
    for key, value in checkpoint.items():
        if hasattr(value, 'shape'):
            if 2048 in value.shape:
                print(f"  {key}: {value.shape} (包含2048)")
                found_2048 = True
    
    if not found_2048:
        print("  未找到2048维度的层")
    
    # 统计参数数量
    total_params = 0
    for key, value in checkpoint.items():
        if hasattr(value, 'numel'):
            total_params += value.numel()
    
    print(f"\n📊 总参数数: {total_params:,}")
    print(f"📊 检查点大小: {sum([v.numel() * 4 for v in checkpoint.values() if hasattr(v, 'numel')]) / 1024**2:.2f} MB")
    
    return checkpoint

def test_model_dimensions():
    """测试模型输入输出维度"""
    try:
        # 添加路径以便导入
        current_dir = os.path.dirname(os.path.abspath(__file__))
        action_folder = os.path.join(current_dir, "action-recognition-pytorch-entropy")
        
        if action_folder not in sys.path:
            sys.path.append(action_folder)
        
        # 尝试导入模型
        try:
            from models.threed_models.degradNet import resnet_degrad
            from models.threed_models.utilityNet import I3Du
            print("成功导入模型")
        except ImportError:
            print("尝试从本地导入...")
            from degradNet import resnet_degrad
            from utilityNet import I3Du
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 测试BDQ模型
        print("\n" + "="*80)
        print("测试BDQ编码器维度")
        print("="*80)
        try:
            bdq_model = resnet_degrad().to(device)
            test_input = torch.randn(2, 3, 16, 224, 224).to(device)
            
            with torch.no_grad():
                output, bias = bdq_model(test_input)
                print(f"输入形状: {test_input.shape}")
                print(f"BDQ输出形状: {output.shape}")
                print(f"BDQ输出展平: {output.view(output.size(0), -1).shape}")
                print(f"偏置形状: {bias.shape}")
        except Exception as e:
            print(f"BDQ模型测试失败: {e}")
        
        # 测试动作识别模型
        print("\n" + "="*80)
        print("测试动作识别模型维度")
        print("="*80)
        try:
            action_model = I3Du(num_classes=8).to(device)
            
            with torch.no_grad():
                action_output = action_model(test_input)
                print(f"动作模型输入形状: {test_input.shape}")
                print(f"动作模型输出形状: {action_output.shape}")
                
                # 检查fc1层权重
                if hasattr(action_model, 'fc1'):
                    print(f"fc1层权重形状: {action_model.fc1.weight.shape}")
                    print(f"期望的输入维度: {action_model.fc1.in_features}")
        except Exception as e:
            print(f"动作模型测试失败: {e}")
            
    except Exception as e:
        print(f"模型维度测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主诊断函数"""
    print("开始诊断...")
    
    # 检查文件存在
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, "action-recognition-pytorch-entropy", "checkpoints", "model_degrad.ckpt")
    
    if os.path.exists(checkpoint_path):
        print(f"找到检查点文件: {checkpoint_path}")
        analyze_checkpoint(checkpoint_path)
    else:
        print(f"检查点文件不存在: {checkpoint_path}")
        print("搜索其他位置...")
        # 尝试其他可能的位置
        possible_paths = [
            "model_degrad.ckpt",
            "checkpoints/model_degrad.ckpt",
            "../model_degrad.ckpt",
            os.path.join(os.path.expanduser("~"), "model_degrad.ckpt")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"找到检查点: {path}")
                analyze_checkpoint(path)
                break
        else:
            print("未找到检查点文件")
    
    # 测试模型维度
    test_model_dimensions()

if __name__ == '__main__':
    main()