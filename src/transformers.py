import random

import torch
from PIL import Image
from torchvision.transforms import transforms
from src.variables import *


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.01):
        """
        mean: Mean of the Gaussian noise.
        std: Standard deviation of the Gaussian noise.
        """
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Adds Gaussian noise to a tensor image.
        Args:
            img (Tensor): Image tensor of shape (C, H, W) with values in [0, 1].
        Returns:
            Tensor: Noisy image.
        """
        if not isinstance(img, torch.Tensor):
            raise TypeError("Input img should be a PyTorch tensor")

        # Add Gaussian noise
        noise = torch.randn_like(img) * self.std + self.mean
        img_noisy = img + noise

        # Clip values to stay within [0, 1]
        img_noisy = torch.clamp(img_noisy, 0.0, 1.0)

        return img_noisy


class AddSaltPepperSpots:
    def __init__(self, salt_prob=0.001, pepper_prob=0.001, spot_size=1):
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob
        self.spot_size = spot_size

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            raise TypeError("Input img should be a PyTorch tensor")

        img_noisy = img.clone()
        _, height, width = img.shape

        num_pixels = height * width
        num_salt_spots = int(num_pixels * random.uniform(0, self.salt_prob))
        num_pepper_spots = int(num_pixels * random.uniform(0, self.pepper_prob))

        # Generate all random top-left coordinates for salt spots
        xs_salt = torch.randint(0, width - self.spot_size + 1, (num_salt_spots,))
        ys_salt = torch.randint(0, height - self.spot_size + 1, (num_salt_spots,))

        for x, y in zip(xs_salt, ys_salt):
            img_noisy[:, y:y + self.spot_size, x:x + self.spot_size] = 1.0

        # Generate all random top-left coordinates for pepper spots
        xs_pepper = torch.randint(0, width - self.spot_size + 1, (num_pepper_spots,))
        ys_pepper = torch.randint(0, height - self.spot_size + 1, (num_pepper_spots,))

        for x, y in zip(xs_pepper, ys_pepper):
            img_noisy[:, y:y + self.spot_size, x:x + self.spot_size] = 0.0

        return img_noisy


downward_img_quality = transforms.Compose([
    transforms.Resize((clip_height // 4, clip_width // 4)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + torch.randn_like(x) * random.uniform(0, 0.03)),
])

normalize_img_size = transforms.Compose([
    transforms.Resize((clip_height, clip_width), Image.BICUBIC),
    transforms.ToTensor()
])

reverse_transform = transforms.Compose([
    transforms.ToPILImage()  # Convert the tensor back to a PIL image
])

to_tensor = transforms.Compose([
    transforms.ToTensor()
])

add_noise = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + torch.randn_like(x) * random.uniform(0, 0.03)),
])
