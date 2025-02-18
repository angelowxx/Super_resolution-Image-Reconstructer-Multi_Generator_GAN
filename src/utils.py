import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


class ImageDatasetWithTransforms(Dataset):
    def __init__(self, folder_path, norm_transform=None, quality_transform=None):
        """
        Args:
            folder_path (str): Path to the folder containing images.
            transform (callable, optional): Transformations to apply to images.
        """
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'jpeg', 'png', 'JPG'))]
        self.norm_transform = norm_transform
        self.quality_transform = quality_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, IOError) as e:
            print(f"Error loading image {img_path}: {e}")
            raise IndexError  # Skip this index

        original_image = self.norm_transform(image) if self.norm_transform else image

        # Apply transformations if defined
        transformed_image = self.quality_transform(image) if self.quality_transform else image

        return original_image, transformed_image


# 定义预处理函数
def get_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])  # 假设输入范围为[-1, 1]
    ])
    return transform


# 将 tensor 转换为 PIL Image
def tensor_to_image(tensor):
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = tensor * 0.5 + 0.5  # 反归一化到 [0, 1]
    tensor = tensor.clamp(0, 1)
    image = transforms.ToPILImage()(tensor)
    return image


# 读取图像并进行预处理
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = get_transform()
    return transform(image).unsqueeze(0)  # 增加 batch 维度


def shuffle_lists_in_same_order(*lists):
    # Combine the lists into a list of tuples
    combined = list(zip(*lists))

    # Sort the combined list based on the last element in each tuple
    combined.sort(key=lambda x: x[-1], reverse=True)

    # Unpack the sorted list back into individual lists
    return [list(t) for t in zip(*combined)]


def interpolate_models(model, target_model, alpha=0.2):
    for param, target_param in zip(model.parameters(), target_model.parameters()):
        param.data = alpha * target_param.data + (1 - alpha) * param.data


"""
用来训练image_encoder
image_encoder 用来取代单纯的像素对比， 待完成
"""


def uniformity_loss(embeddings, t=2):
    # embeddings: Tensor of shape [batch, dim]

    # Check if batch size is 1 (or only one embedding is present)
    if embeddings.size(0) == 1:
        # Return a zero tensor that requires grad, matching the embeddings type and device.
        return torch.zeros(1, device=embeddings.device, dtype=embeddings.dtype, requires_grad=True)

    # Compute pairwise distances (using pdist, which returns distances for each pair)
    pairwise_dists = torch.pdist(embeddings, p=2)

    # Compute loss: encourages large distances
    loss = torch.log(torch.mean(torch.exp(-t * pairwise_dists.pow(2) + 1e-7)))  # Adding small epsilon
    return loss

# 计算 PSNR
def calculate_psnr(img1, img2):
    img1_np = np.array(img1.cpu(), dtype=np.float32)
    img2_np = np.array(img2.cpu(), dtype=np.float32)
    return psnr(img1_np, img2_np, data_range=1)  # 255 是最大像素值

# 计算 SSIM
def calculate_ssim(img1, img2):
    img1_np = np.array(img1.cpu(), dtype=np.float32)
    img2_np = np.array(img2.cpu(), dtype=np.float32)
    return ssim(img1_np, img2_np, data_range=1, multichannel=True, win_size=3)

