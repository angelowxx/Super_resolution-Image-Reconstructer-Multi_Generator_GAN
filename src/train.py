import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import SRResNet, Discriminator
from src.transformers import normalize_img_size, downward_img_quality
from src.utils import ImageDatasetWithTransforms
from PIL import Image
import torchvision.utils as vutils

import torch.nn.functional as F

# 确保结果保存目录存在
os.makedirs("results", exist_ok=True)


def train_example():
    """
    此函数仅为示例，展示如何构建训练循环。
    真实训练过程需要数据集、损失函数、数据加载器等模块。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = [SRResNet().to(device) for i in range(1)]
    discriminator = Discriminator().to(device)
    num_epochs = 50
    optimizer = [optim.Adam(generator.parameters(), lr=0.001) for generator in model]
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    """lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer = optimizer,
        T_max=num_epochs,

    )"""

    image_folder_path = os.path.join(os.getcwd(), 'data', 'train')
    train_data = ImageDatasetWithTransforms(image_folder_path, normalize_img_size, downward_img_quality)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    image_folder_path = os.path.join(os.getcwd(), 'data', 'val')
    val_data = ImageDatasetWithTransforms(image_folder_path, normalize_img_size, downward_img_quality)
    val_loader = DataLoader(val_data, batch_size=5, shuffle=True)

    for epoch in range(num_epochs):
        train_one_epoch(model, discriminator, train_loader, optimizer, d_optimizer, criterion, device, epoch,
                        num_epochs)
        #lr_scheduler.step()

        # 验证：每个epoch结束后随机取一个batch验证效果
        validate(model[-1], val_loader, device, epoch)


def train_one_epoch(model, discriminator, train_loader, g_optimizer, d_optimizer
                    , criterion, device, epoch, num_epochs):
    total_loss = 0
    t = tqdm(train_loader, desc=f"Epoch [{epoch}/{num_epochs}] Training")
    for batch_idx, (hr_imgs, lr_imgs) in enumerate(t):
        hr_imgs = hr_imgs.to(device)
        lr_imgs = lr_imgs.to(device)

        d_loss = train_discriminator(model[0], discriminator, lr_imgs, hr_imgs, criterion, d_optimizer)
        pre_loss = 100
        pre_res = hr_imgs
        g_loss = 0
        for i in range(len(model)):
            generator = model[i]
            optimizer = g_optimizer[i]
            g_loss, sr_imgs = train_generator(generator, discriminator, lr_imgs, hr_imgs,
                                              criterion, optimizer, pre_loss, pre_res)
            pre_loss = g_loss
            pre_res = sr_imgs

        total_loss += g_loss
        t.set_postfix(loss=g_loss)

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch}/{num_epochs}] Training Loss: {avg_loss:.6f}")
    return avg_loss


def train_generator(generator, discriminator, lr_imgs, hr_imgs, criterion, g_optimizer, pre_loss, pre_sr_imgs):
    # --- Train Generator ---
    generator.train()

    sr_images = generator(lr_imgs)

    # Discriminator prediction on fake data
    fake_preds = discriminator(sr_images)

    # Generator loss (fool the discriminator into thinking fake data is real)
    g_loss = criterion(fake_preds, torch.ones_like(fake_preds))

    if g_loss > pre_loss:
        g_loss = criterion(sr_images, pre_sr_imgs)

    else:
        g_loss = g_loss + criterion(sr_images * 5, hr_imgs * 5)

    # Update generator
    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    generator.eval()

    return g_loss.item(), sr_images


def train_discriminator(generator, discriminator, lr_imgs, hr_imgs, criterion, d_optimizer):
    # --- Train Discriminator ---
    discriminator.train()
    # Generate fake data
    sr_images = generator(lr_imgs)

    # Discriminator predictions
    real_preds = discriminator(hr_imgs)
    fake_preds = discriminator(sr_images)

    # Discriminator loss
    d_loss = criterion(real_preds, fake_preds)

    # Update discriminator
    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    discriminator.eval()

    return d_loss.item()


def validate(model, val_loader, device, epoch):
    model.eval()
    with torch.no_grad():
        # 从验证集中获取一个batch
        hr_imgs, lr_imgs = next(iter(val_loader))
        hr_imgs = hr_imgs.to(device)
        lr_imgs = lr_imgs.to(device)
        sr_imgs = model(lr_imgs)
        # 对每个样本拼接：低质量图 | 超分结果 | 原始图
        comp_list = []
        for i in range(hr_imgs.size(0)):
            # 获取原图的尺寸 (高度, 宽度)
            target_size = hr_imgs[i].unsqueeze(0).shape[-2:]
            # 对低质量图像进行上采样，使其尺寸与 hr_imgs[i] 相同
            lr_up = F.interpolate(lr_imgs[i].unsqueeze(0), size=target_size, mode='bilinear',
                                  align_corners=False).squeeze(0)
            comp = torch.cat((lr_up, sr_imgs[i], hr_imgs[i]), dim=2)
            comp_list.append(comp)
        # 制作成图片网格，每行一个样本
        comparison_grid = vutils.make_grid(comp_list, nrow=1, padding=5, normalize=True)
        save_path = os.path.join("results", f"epoch_{epoch}_comparison.png")
        vutils.save_image(comparison_grid, save_path)
        print(f"Epoch {epoch}: Comparison image saved to {save_path}")
    return save_path


if __name__ == "__main__":
    # 如果直接运行 train.py，则调用训练示例
    train_example()
