import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from src.models import VGGFeatureExtractor
from src.transformers import add_noise


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


class ImageDataset(Dataset):
    def __init__(self, folder_path, path1, path2):
        """
        Args:
            folder_path (str): Path to the folder containing images.
            transform (callable, optional): Transformations to apply to images.
        """
        self.folder_path = folder_path

        self.image1_files = [f for f in os.listdir(os.path.join(folder_path, path1)) if
                             f.endswith(('jpg', 'jpeg', 'png', 'JPG'))]
        self.image2_files = [f for f in os.listdir(os.path.join(folder_path, path2)) if
                             f.endswith(('jpg', 'jpeg', 'png', 'JPG'))]
        assert len(self.image1_files) == len(self.image2_files), "the sizes have to be the same!!!"

    def __len__(self):
        return len(self.image1_files)

    def __getitem__(self, idx):
        img1_path = os.path.join(self.folder_path, self.image1_files[idx])
        img2_path = os.path.join(self.folder_path, self.image2_files[idx])
        try:
            image1 = Image.open(img1_path).convert("RGB")
        except (UnidentifiedImageError, IOError) as e:
            print(f"Error loading image {img1_path}: {e}")
            raise IndexError  # Skip this index

        try:
            image2 = Image.open(img2_path).convert("RGB")
        except (UnidentifiedImageError, IOError) as e:
            print(f"Error loading image {img2_path}: {e}")
            raise IndexError  # Skip this index

        return image1, image2


# 将 tensor 转换为 PIL Image
def tensor_to_image(tensor):
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = tensor * 0.5 + 0.5  # 反归一化到 [0, 1]
    tensor = tensor.clamp(0, 1)
    image = transforms.ToPILImage()(tensor)
    return image


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


def perceptal_loss(sr_imgs, hr_imgs, feature_extractor):
    # 使用示例

    features_real = feature_extractor(hr_imgs)
    features_fake = feature_extractor(sr_imgs)

    # 计算感知损失（例如用L1损失）
    perceptual_loss = 0
    l1_loss = torch.nn.L1Loss()
    for key in features_real.keys():
        perceptual_loss += l1_loss(features_fake[key], features_real[key])

    return perceptual_loss


def load_image():
    print()


class ReconstructionLoss(nn.Module):
    def __init__(self):
        """
        Reconstruction loss with edge emphasis.
        :param alpha: Weight for the edge loss component.
        """
        super(ReconstructionLoss, self).__init__()
        self.sobel_x = torch.tensor([[-5, 0, 5],
                                     [-5, 0, 5],
                                     [-5, 0, 5]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.sobel_y = torch.tensor([[-5, -5, -5],
                                     [0, 0, 0],
                                     [5, 5, 5]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.mean_filter = torch.tensor([[1 / 9, 1 / 9, 1 / 9],
                                         [1 / 9, 1 / 9, 1 / 9],
                                         [1 / 9, 1 / 9, 1 / 9]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def normalize(self, x, new_mean=0, new_std=1):
        mean = torch.mean(x)
        std = torch.std(x)
        x = (x - mean) / std
        return x * new_std + new_mean

    def high_pass_filter(self, images):
        # Apply Sobel filters
        sobel_x = self.sobel_x.expand(3, 1, 3, 3).to(images.device)
        sobel_y = self.sobel_y.expand(3, 1, 3, 3).to(images.device)
        mean_filter = self.mean_filter.expand(3, 1, 3, 3).to(images.device)
        edges_x = torch.abs(F.conv2d(images, sobel_x, padding=1, groups=3))  # Horizontal edges
        edges_y = torch.abs(F.conv2d(images, sobel_y, padding=1, groups=3))  # Vertical edges

        # Combine edge maps
        edges = torch.max(edges_x, edges_y)
        for i in range(1):
            edges = F.conv2d(edges, mean_filter, padding=1, groups=3)

        edges = torch.clamp(self.normalize(edges, 0.5, 0.5), 0.1, 0.9)

        return edges

    def total_variation_loss(self, image, reversed_edges):
        # Total Variation Loss (Smoothness penalty)
        diff_i = image[:, :, 1:, 1:] - image[:, :, 1:, :-1]  # Horizontal difference

        diff_j = image[:, :, 1:, 1:] - image[:, :, :-1, 1:]  # Vertical difference

        reversed_edges = reversed_edges[:, :, 1:, 1:].to(image.device)
        diff = (torch.abs(diff_i) + torch.abs(diff_j)) * reversed_edges
        tv_loss = torch.sum(diff) / torch.sum(reversed_edges)

        return tv_loss

    def forward(self, original_images, target_images):
        # L1 loss for pixel-wise similarity
        edges = self.high_pass_filter(original_images)

        reversed_edges = 1 - edges

        diff = torch.abs(original_images - target_images)

        weighted_diff = diff * edges

        # Combine pixel loss and edge loss
        edge_loss = torch.sum(weighted_diff) / torch.sum(edges)
        tv_loss = self.total_variation_loss(target_images, reversed_edges)
        return edge_loss, tv_loss
