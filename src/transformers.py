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
    def __init__(self, salt_prob=0.001, pepper_prob=0.001, spot_size=10):
        """
        salt_prob: Probability of adding 'salt' (white) spots
        pepper_prob: Probability of adding 'pepper' (black) spots
        spot_size: The size of the spots (length of square region in pixels)
        """
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob
        self.spot_size = spot_size

    def __call__(self, img):
        """
        Adds salt-and-pepper spots to a tensor image.
        Args:
            img (Tensor): Image tensor of shape (C, H, W) with values in [0, 1].
        Returns:
            Tensor: Noisy image.
        """
        if not isinstance(img, torch.Tensor):
            raise TypeError("Input img should be a PyTorch tensor")

        # Create a copy of the image
        img_noisy = img.clone()

        _, height, width = img.shape  # Get image dimensions
        num_salt_spots = int(height * width * self.salt_prob)  # Number of salt spots
        num_pepper_spots = int(height * width * self.pepper_prob)  # Number of pepper spots

        """# Add salt (white) spots
        for _ in range(num_salt_spots):
            spot_size = random.randint(1, self.spot_size)
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            img_noisy[:, y:y + spot_size, x:x + spot_size] = 1.0  # Salt is white"""

        # Add pepper (black) spots
        for _ in range(num_pepper_spots):
            spot_size = random.randint(1, self.spot_size)
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            img_noisy[:, y:y + spot_size, x:x + spot_size] = 0.0  # Pepper is black

        return img_noisy


downward_img_quality = transforms.Compose([
    transforms.Resize((clip_height // 4, clip_width // 4)),
    # transforms.Resize((clip_height, clip_width)),
    transforms.ToTensor(),
    #AddGaussianNoise(),
    #AddSaltPepperSpots()
])

normalize_img_size = transforms.Compose([
    transforms.Resize((clip_height, clip_width), Image.BILINEAR),
    transforms.ToTensor()
])

reverse_transform = transforms.Compose([
    transforms.ToPILImage()  # Convert the tensor back to a PIL image
])

to_tensor = transforms.Compose([
    transforms.ToTensor()
])
