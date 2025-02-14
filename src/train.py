import os

import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import SRResNet, Discriminator
from src.transformers import normalize_img_size, downward_img_quality
from src.utils import ImageDatasetWithTransforms, shuffle_lists_in_same_order
from PIL import Image
import torchvision.utils as vutils

import torch.nn.functional as F


def train_example(num_epochs, num_models):
    """
    此函数仅为示例，展示如何构建训练循环。
    真实训练过程需要数据集、损失函数、数据加载器等模块。
    """
    # 确保结果保存目录存在
    os.makedirs(f"results{num_models}", exist_ok=True)
    os.makedirs(f'figures{num_models}', exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = torch.nn.MSELoss()
    discriminator = Discriminator().to(device)
    model = [SRResNet().to(device) for i in range(num_models)]
    optimizer = [optim.Adam(generator.parameters(), lr=0.001) for generator in model]
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
    d_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=d_optimizer,
        step_size=5,
        gamma=0.6
    )
    lr_schedulers = []
    for i in range(len(model)):
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer[i],
            step_size=5,
            gamma=0.6
        )
        lr_schedulers.append(scheduler)

    image_folder_path = os.path.join(os.getcwd(), 'data', 'train')
    train_data = ImageDatasetWithTransforms(image_folder_path, normalize_img_size, downward_img_quality)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    image_folder_path = os.path.join(os.getcwd(), 'data', 'val')
    val_data = ImageDatasetWithTransforms(image_folder_path, normalize_img_size, downward_img_quality)
    val_loader = DataLoader(val_data, batch_size=5, shuffle=True)

    avg_losses = []

    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, discriminator, train_loader, optimizer, d_optimizer, criterion, device, epoch,
                                   num_epochs)
        avg_losses.append(avg_loss)

        for scheduler in lr_schedulers:
            scheduler.step()

        d_lr_scheduler.step()

        # 验证：每个epoch结束后随机取一个batch验证效果
        validate(model[-1], val_loader, device, epoch, num_models)

        shuffle_lists_in_same_order(model, lr_schedulers, optimizer)

    # Save the generator model's state_dict
    avg_losses[0] = 2 # 防止第一个损失太大带来的曲线偏离，无法看清后续的变化趋势
    for i in range(len(model)):
        torch.save(model[i].state_dict(), os.path.join(f'results{num_models}', f'generator_model_{i}.pth'))
    # Plotting the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), avg_losses, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot as an image file in the 'figures' directory
    plt.savefig(os.path.join(f'figures{num_models}', 'training_loss_curve.png'))


def train_one_epoch(model, discriminator, train_loader, g_optimizer, d_optimizer
                    , criterion, device, epoch, num_epochs):
    total_loss = 0
    t = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Training")
    for batch_idx, (hr_imgs, lr_imgs) in enumerate(t):
        hr_imgs = hr_imgs.to(device)
        lr_imgs = lr_imgs.to(device)

        d_loss = train_discriminator(model[0], discriminator, lr_imgs, hr_imgs, criterion, d_optimizer)
        pre_loss = 100
        pre_res = hr_imgs
        first_loss = 0
        for i in range(len(model)):
            generator = model[i]
            optimizer = g_optimizer[i]
            g_loss, sr_imgs = train_generator(generator, discriminator, lr_imgs, hr_imgs,
                                              criterion, optimizer, pre_loss, pre_res)
            pre_loss = g_loss
            pre_res = sr_imgs
            if i == 0:
                first_loss = g_loss

        total_loss += first_loss
        t.set_postfix(loss=first_loss)

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_loss:.6f}")
    return avg_loss


def train_generator(generator, discriminator, lr_imgs, hr_imgs, criterion, g_optimizer, pre_loss, pre_sr_imgs):
    # --- Train Generator ---
    generator.train()

    sr_images = generator(lr_imgs)

    # Discriminator prediction on fake data
    fake_preds = discriminator(sr_images)

    # Generator loss (fool the discriminator into thinking fake data is real)
    g_loss = criterion(fake_preds, torch.ones_like(fake_preds))

    # 当前loss比pre_loss大时，当前generator向前一个学习
    # 或者改成按概率决定 sigma = Norm(g_loss, pre_loss**2), if sigma > pre_loss
    sigma = torch.normal(mean=g_loss, std=g_loss ** 2)  # 生成 sigma
    if sigma > pre_loss:
        g_loss = criterion(sr_images, pre_sr_imgs.detach())

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


def validate(model, val_loader, device, epoch, num_models):
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
        save_path = os.path.join(f"results{num_models}", f"epoch_{epoch+1}_comparison.png")
        vutils.save_image(comparison_grid, save_path)
        print(f"Epoch {epoch}: Comparison image saved to {save_path}")
    return save_path


if __name__ == "__main__":
    # 如果直接运行 train.py，则调用训练示例
    train_example(40, 1)
    train_example(40, 3)
