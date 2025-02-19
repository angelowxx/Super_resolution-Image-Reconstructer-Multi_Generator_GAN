import os
import random

import torch.multiprocessing as mp
import torch.distributed as dist

import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, random_split
from tqdm import tqdm

from src.models import SRResNet, ImageFingerPrint, Discriminator
from src.transformers import normalize_img_size, downward_img_quality
from src.utils import ImageDatasetWithTransforms, shuffle_lists_in_same_order, interpolate_models, \
    uniformity_loss, calculate_psnr, calculate_ssim
from PIL import Image
import torchvision.utils as vutils

import torch.nn.functional as F

nums_epoch = 30


def train_example(rank, world_size, num_epochs):
    # 初始化进程组
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'  # 选择一个未被占用的端口
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # 设置 GPU
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # 确保结果保存目录存在
    os.makedirs(f"results", exist_ok=True)

    lr_generator = 2e-4
    lr_image_finger_print = 2e-4
    lr_dicriminator = 2e-4

    g_criterion = torch.nn.L1Loss()

    image_finger_print = ImageFingerPrint().to(device)
    image_finger_print = nn.SyncBatchNorm.convert_sync_batchnorm(image_finger_print)
    image_finger_print = nn.parallel.DistributedDataParallel(image_finger_print, device_ids=[rank])

    generator = nn.parallel.DistributedDataParallel(SRResNet().to(device), device_ids=[rank])

    discriminator = nn.parallel.DistributedDataParallel(Discriminator().to(device), device_ids=[rank])

    g_optimizer = optim.Adam(generator.parameters(), lr=lr_generator)
    image_fingerprint_optimizer = optim.Adam(image_finger_print.parameters(), lr=lr_image_finger_print)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr_dicriminator)

    scheduler = optim.lr_scheduler.CosineAnnealingLR

    i_lr_scheduler = scheduler(
        optimizer=image_fingerprint_optimizer,
        T_max=num_epochs
    )
    lr_scheduler = scheduler(
        optimizer=g_optimizer,
        T_max=num_epochs
    )
    d_lr_scheduler = scheduler(
        optimizer=d_optimizer,
        T_max=num_epochs
    )

    # Define paths
    train_folder_path = os.path.join(os.getcwd(), 'data', 'train')

    # Load full dataset
    train_data = ImageDatasetWithTransforms(train_folder_path, normalize_img_size, downward_img_quality)

    # Define split sizes (e.g., 70% train, 30% validation)
    split_ratio = 0.8
    train_size = int(split_ratio * len(train_data))
    val_size = len(train_data) - train_size

    # Split dataset into two parts
    train_subset, val_subset = random_split(train_data, [train_size, val_size])

    # Create samplers
    train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_subset, num_replicas=world_size, rank=rank)

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=8, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=8, sampler=val_sampler, num_workers=0)

    psnrs = []
    ssims = []

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)  # 保证不同 GPU 训练的数据不重复
        val_sampler.set_epoch(epoch)

        # if epoch > -1:
        #    g_criterion = PerceptualLoss(device=device)# 内存不够，以后再说
        train_one_epoch(generator, image_finger_print, train_loader, g_optimizer, image_fingerprint_optimizer
                        , g_criterion, device, epoch, num_epochs, discriminator, d_optimizer)

        i_lr_scheduler.step()

        lr_scheduler.step()

        d_lr_scheduler.step()

        # 验证：每个epoch结束后随机取一个batch验证效果
        if (epoch+1) % 5 == 0 :
            validate(generator, train_loader, device, epoch, "fingerprint", dist.get_rank())

        psnr, ssim = compute_score(generator, train_loader, device)
        psnrs.append(psnr / 30)
        ssims.append(ssim)

    # Save the generator model's state_dict
    torch.save(generator.state_dict(), os.path.join(f'results', f'generator_model_{dist.get_rank()}.pth'))
    # Plotting the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), psnrs, marker='o', linestyle='-', color='b', label='PNSR/30')
    plt.plot(range(1, num_epochs + 1), ssims, marker='o', linestyle='--', color='r', label='SSIM')
    plt.title('Rating Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Rating Value')
    plt.legend()
    plt.grid(True)

    # Save the plot as an image file in the 'results' directory
    plt.savefig(os.path.join(f'results', f'training_loss_curve_{dist.get_rank()}.png'))

    dist.destroy_process_group()  # 训练结束后销毁进程组


def train_one_epoch(generator, image_finger_print, train_loader, g_optimizer, image_fingerprint_optimizer
                    , g_criterion, device, epoch, num_epochs, discriminator, d_optimizer):
    total_loss = 0
    description = "Training"
    t = tqdm(train_loader, desc=f"[{epoch + 1}/{num_epochs}] {description}")
    for batch_idx, (hr_imgs, lr_imgs) in enumerate(t):
        hr_imgs = hr_imgs.to(device)
        lr_imgs = lr_imgs.to(device)

        # i_loss = train_image_finger_print(image_finger_print, hr_imgs, image_fingerprint_optimizer)

        d_loss = train_discriminator(discriminator, generator, hr_imgs, lr_imgs, d_optimizer)

        g_loss = train_generator(generator, image_finger_print, discriminator, lr_imgs, hr_imgs,
                                 g_criterion, g_optimizer)

        t.set_postfix(g=g_loss, d=d_loss)
        total_loss += g_loss

    avg_loss = total_loss / len(train_loader)

    print(f"Epoch [{epoch + 1}/{num_epochs}] {description} Loss: {avg_loss:.6f}")
    return avg_loss


def train_generator(generator, image_finger_print, discriminator, lr_imgs, hr_imgs,
                    g_criterion, g_optimizer):
    torch.autograd.set_detect_anomaly(True)
    # --- Train Generator ---
    generator.train()
    image_finger_print.eval()
    discriminator.eval()

    sr_images = generator(lr_imgs)

    fake_prints = image_finger_print(sr_images)
    fake_preds = discriminator(sr_images)

    with torch.no_grad():
        real_prints = image_finger_print(hr_imgs)
        real_preds = discriminator(hr_imgs)

    g_loss = g_criterion(sr_images, hr_imgs) \
             + torch.mean(torch.tanh(real_preds - fake_preds))
    # g_loss = torch.mean(real_preds-fake_preds)
    # g_loss = g_criterion(fake_prints, real_prints)
    # g_loss = g_criterion(sr_images, hr_imgs)

    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    loss_item = g_loss.item()

    del g_loss
    torch.cuda.empty_cache()  # Free unused memory

    return loss_item


def train_image_finger_print(image_finger_print, hr_imgs, image_fingerprint_optimizer):
    torch.autograd.set_detect_anomaly(True)
    # --- Train image_finger_print ---
    image_finger_print.train()

    # Get image_finger_print predictions
    real_preds = image_finger_print(hr_imgs)

    d_loss = uniformity_loss(real_preds)

    # Update image_finger_print
    image_fingerprint_optimizer.zero_grad()
    d_loss.backward()
    image_fingerprint_optimizer.step()

    loss_item = d_loss.item()

    del d_loss  # Delete loss tensor
    torch.cuda.empty_cache()  # Free unused memory

    return loss_item


def train_discriminator(discriminator, generator, hr_imgs, lr_imgs, d_optimizer):
    torch.autograd.set_detect_anomaly(True)
    # --- Train image_finger_print ---
    discriminator.train()
    generator.eval()

    sr_imgs = generator(lr_imgs)

    # Get image_finger_print predictions
    real_preds = discriminator(hr_imgs)
    fake_preds = discriminator(sr_imgs)

    d_loss = torch.mean(torch.tanh(fake_preds - real_preds))

    # Update image_finger_print
    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    loss_item = d_loss.item()

    del d_loss  # Delete loss tensor
    torch.cuda.empty_cache()  # Free unused memory

    return loss_item


def validate(model, val_loader, device, epoch, desc, rank):
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
        save_path = os.path.join(f"results", f"{desc}_epoch_{epoch + 1}_{rank}_comparison.png")
        vutils.save_image(comparison_grid, save_path)
        print(f"Epoch {epoch + 1} rank{rank}: Comparison image saved to {save_path}")

    return save_path


def compute_score(model, val_loader, device):
    model.eval()
    with torch.no_grad():
        psnr = 0
        ssim = 0
        # 从验证集中获取一个batch
        hr_imgs, lr_imgs = next(iter(val_loader))
        hr_imgs = hr_imgs.to(device)
        lr_imgs = lr_imgs.to(device)
        sr_imgs = model(lr_imgs)

        for i in range(hr_imgs.size(0)):
            psnr += calculate_psnr(sr_imgs[i], hr_imgs[i])
            ssim += calculate_ssim(sr_imgs[i], hr_imgs[i])

        psnr /= hr_imgs.size(0)
        ssim /= hr_imgs.size(0)
        print(f'psnr={psnr}, ssim={ssim}')

    return psnr, ssim


if __name__ == "__main__":
    print(f'Training!')

    world_size = torch.cuda.device_count()
    mp.spawn(train_example, args=(world_size, nums_epoch), nprocs=world_size)
