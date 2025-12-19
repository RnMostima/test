import torch
from models.model_builder import build_model  # 导入模型构建函数
import argparse

# 1. 配置模型参数（需与预训练模型的训练配置一致）
args = argparse.Namespace(
    # 已有的参数
    modality='rgb',
    input_channels=3,
    threed_data=True,
    num_classes=400,
    batch_size=4,
    workers=4,
    datadir='path/to/dataset',  # 数据集路径
    input_size=224,
    backbone_net='i3d',  # 模型名称（如之前补充的i3d）
    dropout=0.5,  # 之前补充的dropout参数
    without_t_stride=False,  # 之前补充的参数
    pooling_method='avg',  # 之前补充的参数
    
    # 补充dataset参数（关键）
    dataset='kinetics400',  # 数据集名称，常见为'kinetics400'（与预训练模型匹配）
    
    # 补充其他可能需要的参数（参考test.py）
    groups=64,  # 分组数，test.py中提到'arch_name'包含'f64'，对应groups=64
    frames_per_group=1,  # 每组帧数，模型构建可能需要
)

# 2. 构建模型（test_mode=True表示推理模式，不初始化未使用的层）
model, arch_name = build_model(args, test_mode=True)
model_degrad = model[0]  # 提取BDQ编码器（模型列表的第一个元素）

# 3. 加载预训练权重
ckpt_path = "checkpoints/model_degrad.ckpt"  # 预训练模型路径
checkpoint = torch.load(ckpt_path, map_location='cpu')  # 加载到CPU，避免设备不匹配

# 4. 处理state_dict（去除训练时可能的module.前缀）
# 若训练时使用了DataParallel/DistributedDataParallel，权重键会带module.，需移除
state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}

# 5. 加载权重到模型
model_degrad.load_state_dict(state_dict)  # 若结构完全匹配，用strict=True（默认）
# 若有少量不匹配的键（如训练时额外保存的参数），可设strict=False：
# model_degrad.load_state_dict(state_dict, strict=False)

# 6. 设置为评估模式（关键：推理时禁用Dropout、BatchNorm更新）
model_degrad.eval()

# （可选）若有GPU，转移模型到GPU
if torch.cuda.is_available():
    model_degrad = model_degrad.cuda()