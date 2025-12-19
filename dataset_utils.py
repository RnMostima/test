# dataset_utils.py
import os
import glob
import cv2
import torch
import torch.nn.functional as F
from torchvision.models import resnet50
from torch.utils.data import Dataset  # 改用基础Dataset类，不再依赖VideoDataSet

# 隐私属性识别模型
class PrivacyModel(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, num_classes)
    def forward(self, x):
        # 适配3D输入（B,C,T,H,W）→ 取中间帧转2D（B,C,H,W）
        if len(x.shape) == 5:
            x = x[:, :, x.shape[2]//2, :, :]
        return self.resnet(x)

# 自定义SBU数据集类（改用基础Dataset，手动处理所有逻辑，彻底避开VideoRecord）
class SBUDataSet(Dataset):
    def __init__(self, root, list_file, num_groups, frames_per_group, transform, is_train, image_tmpl):
        super().__init__()
        self.root = root  # 数据集根路径
        self.num_groups = num_groups  # 分组数（兼容原参数）
        self.frames_per_group = frames_per_group  # 每组帧数（兼容原参数）
        self.transform = transform  # 数据增强
        self.is_train = is_train  # 训练/验证模式
        self.image_tmpl = image_tmpl  # 帧命名模板（如：img_%05d.jpg）
        self.label_list = self._parse_list_file(list_file)  # 手动解析标注文件

    def _parse_list_file(self, list_file):
        """手动解析标注文件，返回[(video_path, frame_cnt, action_label, privacy_label), ...]"""
        label_list = []
        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 拆分标注行（兼容4字段：路径,帧数,动作标签,隐私标签 | 5字段：路径,xxx,动作标签,隐私标签,xxx）
                parts = line.split(",")
                if len(parts) >= 4:
                    # 处理4字段：path, frame_cnt, action, privacy
                    if len(parts) == 4:
                        video_path = parts[0]
                        frame_cnt = int(parts[1])
                        action_label = int(parts[2])
                        privacy_label = int(parts[3])
                    # 处理5字段：path, xxx, action, privacy, xxx
                    else:
                        video_path = parts[0]
                        frame_cnt = self._get_actual_frame_count(video_path)  # 自动统计帧数
                        action_label = int(parts[2])
                        privacy_label = int(parts[3])
                    label_list.append((video_path, frame_cnt, action_label, privacy_label))
        return label_list

    def _get_actual_frame_count(self, video_path):
        """统计视频帧文件夹的实际帧数"""
        frame_pattern = os.path.join(self.root, video_path, self.image_tmpl.replace("%05d", "*"))
        frame_files = glob.glob(frame_pattern)
        return len(frame_files) if frame_files else 64  # 无帧时默认64帧

    def _load_frames(self, video_path, frame_cnt):
        """加载视频帧（采样固定帧数，兼容不足的情况）"""
        frame_pattern = os.path.join(self.root, video_path, self.image_tmpl)
        frame_files = sorted(glob.glob(frame_pattern.replace("%05d", "*")))
        if not frame_files:
            # 无帧时返回空白图像（64帧，224x224x3）
            return [torch.zeros(3, 224, 224) for _ in range(64)]
        # 采样帧数（均匀采样）
        sample_indices = self._sample_indices(len(frame_files), frame_cnt)
        frames = []
        for idx in sample_indices:
            img = cv2.imread(frame_files[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR→RGB
            if self.transform:
                img = self.transform(img)  # 应用增强（转为tensor）
            else:
                img = torch.from_numpy(img).permute(2, 0, 1) / 255.0  # HWC→CHW，归一化
            frames.append(img)
        # 不足帧数时补最后一帧
        while len(frames) < frame_cnt:
            frames.append(frames[-1])
        return frames

    def _sample_indices(self, total_frames, target_frames):
        """均匀采样索引（兼容训练/验证模式，这里简化为均匀采样）"""
        if total_frames <= target_frames:
            return list(range(total_frames))
        # 均匀间隔采样
        step = total_frames / target_frames
        indices = [int(i * step) for i in range(target_frames)]
        return indices

    def __len__(self):
        """数据集长度"""
        return len(self.label_list)

    def __getitem__(self, idx):
        """核心：返回(frames, action_label, privacy_label)"""
        # 从手动解析的列表中获取数据（彻底避开VideoRecord）
        video_path, frame_cnt, action_label, privacy_label = self.label_list[idx]
        # 修正帧数（用实际帧数覆盖标注的无效帧数）
        frame_cnt = self._get_actual_frame_count(video_path) or frame_cnt
        # 加载并采样帧
        frames = self._load_frames(video_path, frame_cnt)
        # 拼接为3D张量（T,C,H,W）→ (C,T,H,W)（适配3D模型）
        frames = torch.stack(frames).permute(1, 0, 2, 3)
        return frames, action_label, privacy_label

# 训练工具函数
# dataset_utils.py 中的 train_task 函数（无需修改，保留即可）
def train_task(model_degrad, model_task, loader, criterion, optimizer, task_type, device, accumulate_steps=4):
    model_degrad.train()
    model_task.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    # ========== 新增：提取model_task的前几层（conv1层，处理3通道输入） ==========
    # 方法1：手动截取到conv1层（若知道模型结构）
    class ModelFront(torch.nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.conv1 = original_model.conv1  # 保留conv1层（3→64）
            # 若有bn1、relu等，也保留：self.bn1 = original_model.bn1; self.relu = original_model.relu
        
        def forward(self, x):
            x = self.conv1(x)
            # x = self.bn1(x)
            # x = self.relu(x)
            return x

    model_front = ModelFront(model_task).to(device)  # 仅保留前几层处理3通道

    for idx, (frames, action_lab, privacy_lab) in enumerate(loader):
        torch.cuda.empty_cache()
        frames = frames.to(device, non_blocking=True)
        labels = action_lab.to(device, non_blocking=True) if task_type == "action" else privacy_lab.to(device, non_blocking=True)

        # ========== 步骤1：3通道输入先过model_task的前几层（3→64） ==========
        feature_64 = model_front(frames)  # 输出：[B,64,D,H,W]

        # ========== 步骤2：64通道特征过BDQ编码器（→8192维） ==========
        processed_frames = model_degrad(feature_64)  # BDQ编码器处理64通道特征
        if isinstance(processed_frames, tuple):
            temp_tuple = processed_frames
            processed_frames = temp_tuple[0]
            del temp_tuple

        # ========== 步骤3：维度调配（8192→2048，回到之前的逻辑） ==========
        # 此时processed_frames是8192维，调用之前的维度调配函数（处理8192→2048，保留高维结构）
        processed_frames_adjusted = adjust_feature_dim(processed_frames, target_dim=2048)

        # ========== 步骤4：processed_frames_adjusted传入model_task的后续层 ==========
        # 方法1：若model_task可截取后续层，继续处理；若不可，直接用该特征做分类（简化版）
        # 简化版：添加一个全连接层，2048→类别数（如8类）
        fc = torch.nn.Linear(2048, 8).to(device)  # 假设类别数是8
        # 展平特征：[B,2048,D,H,W] → [B,2048*D*H*W]（若维度不对，调整为[B,2048]）
        feature_flat = processed_frames_adjusted.view(processed_frames_adjusted.size(0), -1)
        # 若特征维度超过2048，池化到2048维：feature_flat = F.adaptive_avg_pool1d(feature_flat.unsqueeze(1), 2048).squeeze(1)
        outputs = fc(feature_flat[:, :2048])  # 取前2048维做分类

        # ========== 后续逻辑不变 ==========
        loss = criterion(outputs, labels) / accumulate_steps
        loss.backward()
        total_loss += loss.item() * accumulate_steps

        if (idx + 1) % accumulate_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        del frames, feature_64, processed_frames, processed_frames_adjusted, outputs, loss, labels
        torch.cuda.empty_cache()

    if idx % accumulate_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / len(loader)


def test_task(model_degrad, model_task, loader, criterion, use_bdq, task_type, device):
    model_degrad.eval()
    model_task.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for frames, action_lab, privacy_lab in loader:
            frames = frames.to(device)
            labels = action_lab.to(device) if task_type == "action" else privacy_lab.to(device)
            
            if use_bdq:
                processed_frames = model_degrad(frames)
                if isinstance(processed_frames, tuple):
                    processed_frames = processed_frames[0]
                # 调用新的维度调配函数（保留高维结构）
                frames = adjust_feature_dim(processed_frames, target_dim=2048)
            else:
                frames = adjust_feature_dim(frames, target_dim=2048)
            
            outputs = model_task(frames)
            loss = criterion(outputs, labels)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            total_loss += loss.item()
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(loader)
    return accuracy, avg_loss

# 张量转图像工具函数（适配不同维度的张量）
def tensor2img(tensor, mean, std):
    """将模型输出张量（C,T,H,W/C,H,W）转为CV2可显示图像"""
    # 处理3D张量（C,T,H,W）→ 2D张量（C,H,W）（取中间帧）
    if len(tensor.shape) == 4:
        tensor = tensor[:, tensor.shape[1]//2, :, :]
    # 维度转换：(C,H,W) → (H,W,C)
    img = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    # 反归一化（恢复像素值范围）
    img = img * np.array(std) + np.array(mean)
    # 限制像素值在0-1之间
    img = np.clip(img, 0, 1)
    # 转为8位整数（CV2显示格式）
    return (img * 255).astype(np.uint8)

# 提取BDQ三个阶段输出的工具函数（兼容不同模型结构）
def get_bdq_stages(model_degrad, input_frames):
    """获取BDQ处理的三个阶段输出：高斯平滑→帧差→量化输出"""
    model_degrad.eval()
    with torch.no_grad():
        # 阶段1：高斯平滑（兼容模型自有gauss层或通用平均池化）
        if hasattr(model_degrad, 'gauss'):
            stage1 = model_degrad.gauss(input_frames)
        else:
            # 3D平均池化实现高斯平滑（核大小1x3x3，填充0x1x1）
            stage1 = torch.nn.functional.avg_pool3d(input_frames, kernel_size=(1,3,3), padding=(0,1,1))
        # 阶段2：帧差计算（当前帧 - 前一帧，去除第一帧的NaN）
        stage2 = stage1 - torch.roll(stage1, 1, dims=2)[:, :, 1:, ...]
        # 阶段3：完整BDQ处理后的输出
        stage3 = model_degrad(input_frames)
        if isinstance(stage3, tuple):
            stage3 = stage3[0]
    return stage1, stage2, stage3

def adjust_feature_dim(feature, target_dim=2048):
    """
    维度调配：3通道高维张量 → 2048维（8G GPU可承载）
    核心：1. 3D池化压缩时空维度 2. 分阶段升维（3→256→2048） 3. 全程低显存
    """
    # 强制清理CUDA缓存（第一步）
    torch.cuda.empty_cache()

    batch_size = feature.size(0)
    input_shape = feature.shape
    input_c = input_shape[1]  # 3
    dims = len(input_shape)     # 输入维度数

    # ========== 步骤1：补全为5D张量（[B,3,D,H,W]） ==========
    while len(feature.shape) < 5:
        feature = feature.unsqueeze(-1)
    feature_5d = feature  # 形状：[B,3,D,H,W]

    # ========== 步骤2：极致压缩时空维度（3D池化，核心！减少元素数量） ==========
    # 方案1：固定核3D池化（压缩D/H/W到1/4）
    # 池化核：(2,4,4) → D压缩到1/2，H/W压缩到1/4（可根据需要调大，如(4,8,8)）
    feature_pooled = F.avg_pool3d(feature_5d, kernel_size=(2,4,4), stride=(2,4,4), padding=0)
    # 方案2：自适应3D池化（强制压缩到固定小尺寸，如D=4, H=16, W=16，推荐！）
    # feature_pooled = F.adaptive_avg_pool3d(feature_5d, output_size=(4, 16, 16))  # 时空维度固定为小尺寸

    # ========== 步骤3：分阶段升维（3→256→2048，避免单次升维显存爆炸） ==========
    # 阶段1：3→256（1x1x1 3D卷积，参数：3*256=768，显存可忽略）
    conv1 = torch.nn.Conv3d(3, 256, kernel_size=1, bias=False).to(feature.device)
    feature_256 = conv1(feature_pooled)
    feature_256 = F.relu(feature_256)  # 激活函数，增加非线性（可选）
    del conv1, feature_pooled  # 释放显存

    # 阶段2：256→2048（1x1x1 3D卷积，参数：256*2048=524,288，显存仍可承载）
    conv2 = torch.nn.Conv3d(256, target_dim, kernel_size=1, bias=False).to(feature.device)
    adjusted_5d = conv2(feature_256)
    del conv2, feature_256  # 释放显存

    # ========== 步骤4：恢复为输入的维度数 ==========
    while len(adjusted_5d.shape) > dims:
        adjusted_5d = adjusted_5d.squeeze(-1)
    adjusted_feature = adjusted_5d

    # 最终清理显存
    del feature_5d
    torch.cuda.empty_cache()

    return adjusted_feature