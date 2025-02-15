import os

import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import SRResNet, Discriminator, PerceptualLoss
from src.transformers import normalize_img_size, downward_img_quality
from src.utils import ImageDatasetWithTransforms, shuffle_lists_in_same_order
from PIL import Image
import torchvision.utils as vutils

import torch.nn.functional as F


def train_example(num_epochs, num_models):
    print(f'Training with {num_models} generators competing!')
    # 确保结果保存目录存在
    os.makedirs(f"results{num_models}", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d_criterion = torch.nn.L1Loss()
    g_criterion = torch.nn.L1Loss()
    discriminator = Discriminator().to(device)
    model = [SRResNet().to(device) for i in range(num_models)]
    optimizer = [optim.Adam(generator.parameters(), lr=2e-4) for generator in model]
    d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR
    d_lr_scheduler = scheduler(
        optimizer=d_optimizer,
        T_max=num_epochs,
    )
    lr_schedulers = []
    for i in range(len(model)):
        g_lr_scheduler = scheduler(
            optimizer=optimizer[i],
            T_max=num_epochs,
        )
        lr_schedulers.append(g_lr_scheduler)

    image_folder_path = os.path.join(os.getcwd(), 'data', 'train')
    train_data = ImageDatasetWithTransforms(image_folder_path, normalize_img_size, downward_img_quality)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    image_folder_path = os.path.join(os.getcwd(), 'data', 'val')
    val_data = ImageDatasetWithTransforms(image_folder_path, normalize_img_size, downward_img_quality)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=True)

    avg_losses = []

    for epoch in range(num_epochs):
        #if epoch > -1:
        #    g_criterion = PerceptualLoss(device=device)# 内存不够，以后再说
        gen_losses = [0 for i in range(len(model))]
        avg_loss = train_one_epoch(model, discriminator, train_loader, optimizer, d_optimizer, d_criterion, g_criterion,
                                   device, epoch, num_epochs, gen_losses)
        avg_losses.append(avg_loss)

        for scheduler in lr_schedulers:
            scheduler.step()

        d_lr_scheduler.step()

        # 验证：每个epoch结束后随机取一个batch验证效果
        validate(model[-1], val_loader, device, epoch, num_models)

        shuffle_lists_in_same_order(model, lr_schedulers, optimizer, gen_losses)

    # Save the generator model's state_dict
    #avg_losses[0] = 1 # 防止第一个损失太大带来的曲线偏离，无法看清后续的变化趋势
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

    # Save the plot as an image file in the 'results' directory
    plt.savefig(os.path.join(f'results{num_models}', 'training_loss_curve.png'))


def train_one_epoch(model, discriminator, train_loader, g_optimizer, d_optimizer
                    , d_criterion, g_criterion, device, epoch, num_epochs, gen_losses):
    total_loss = 0
    t = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Training")
    for batch_idx, (hr_imgs, lr_imgs) in enumerate(t):
        hr_imgs = hr_imgs.to(device)
        lr_imgs = lr_imgs.to(device)

        d_loss = train_discriminator(model[0], discriminator, lr_imgs, hr_imgs, d_criterion, d_optimizer)
        pre_loss = 100
        pre_res = hr_imgs
        first_loss = 0
        for i in range(len(model)):
            generator = model[i]
            optimizer = g_optimizer[i]
            g_loss, sr_imgs = train_generator(generator, discriminator, lr_imgs, hr_imgs,
                                              g_criterion, optimizer, pre_loss, pre_res)
            if g_loss < pre_loss:
                pre_res = sr_imgs
                pre_loss = g_loss
            if i == 0:
                first_loss = g_loss
            gen_losses[i] += g_loss

        total_loss += first_loss
        t.set_postfix(g_loss=first_loss, d_loss=d_loss)

    avg_loss = total_loss / len(train_loader)

    for i in range(len(gen_losses)):
        gen_losses[i] /= len(train_loader)  # 直接修改原列表

    print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_loss:.6f}")
    return avg_loss


def train_generator(generator, discriminator, lr_imgs, hr_imgs,
                    criterion, g_optimizer, pre_loss, pre_sr_imgs):
    # --- Train Generator ---
    generator.train()

    sr_images = generator(lr_imgs)

    com_loss = criterion(sr_images, hr_imgs)
    g_loss = com_loss

    # Discriminator prediction on fake data
    fake_preds = discriminator(sr_images)

    # 当前loss比pre_loss大时，当前generator向前一个学习
    # 或者改成按概率决定 sigma = Norm(g_loss, pre_loss**2), if sigma > pre_loss
    theta = abs(g_loss.item()-pre_loss)
    sigma = torch.normal(mean=g_loss, std=theta ** 2)  # 生成 sigma
    if sigma > pre_loss:
        g_loss = g_loss + criterion(sr_images, pre_sr_imgs.detach())

    else:
        g_loss = g_loss + criterion(fake_preds, torch.ones_like(fake_preds))

    # Update generator
    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    generator.eval()

    return com_loss.item(), sr_images


def train_discriminator(generator, discriminator, lr_imgs, hr_imgs, criterion, d_optimizer):
    # --- Train Discriminator ---
    discriminator.train()

    # Generate fake images
    with torch.no_grad():  # 不让生成器回传梯度
        sr_images = generator(lr_imgs)

    # Get discriminator predictions
    real_preds = discriminator(hr_imgs)
    fake_preds = discriminator(sr_images.detach())  

    # Create real and fake labels
    real_labels = torch.ones_like(real_preds)
    fake_labels = torch.zeros_like(fake_preds)

    # Compute losses separately
    real_loss = criterion(real_preds, real_labels)
    fake_loss = criterion(fake_preds, fake_labels)

    # Total discriminator loss
    d_loss = (real_loss + fake_loss) / 2  # 平均损失

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
        comparison_grid = vutils.make_grid(comp_list, nrow=1, padding=5, normalize=False)
        save_path = os.path.join(f"results{num_models}", f"epoch_{epoch+1}_comparison.png")
        vutils.save_image(comparison_grid, save_path)
        print(f"Epoch {epoch+1}: Comparison image saved to {save_path}")
    return save_path


if __name__ == "__main__":

    train_example(50, 3)
    train_example(50, 1)
