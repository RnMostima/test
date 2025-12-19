"""
配置文件：确保所有路径正确
"""

import os

# 基础路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据集路径
SBU_DATA_DIR = os.path.join(ROOT_DIR, "SBU")
TRAIN_TXT_PATH = os.path.join(ROOT_DIR, "train.txt")
VAL_TXT_PATH = os.path.join(ROOT_DIR, "val.txt")

# 模型路径
ACTION_FOLDER = os.path.join(ROOT_DIR, "action-recognition-pytorch-entropy")
CKPT_PATH = os.path.join(ACTION_FOLDER, "checkpoints", "model_degrad.ckpt")

# 输出路径
VIS_DIR = os.path.join(ROOT_DIR, "visualization")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# 创建必要的目录
for dir_path in [VIS_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 验证路径
print("路径验证:")
print(f"根目录: {ROOT_DIR}")
print(f"SBU数据目录: {os.path.exists(SBU_DATA_DIR)}")
print(f"训练文件: {os.path.exists(TRAIN_TXT_PATH)}")
print(f"验证文件: {os.path.exists(VAL_TXT_PATH)}")
print(f"模型文件: {os.path.exists(CKPT_PATH)}")