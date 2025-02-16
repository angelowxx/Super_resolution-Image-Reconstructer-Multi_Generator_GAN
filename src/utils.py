import os
import random

import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset


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


def interpolate_models(model, target_model, alpha=1-1e-5):
    for param, target_param in zip(model.parameters(), target_model.parameters()):
        param.data = alpha * target_param.data + (1 - alpha) * param.data



