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

from src.models import SRResNet, Discriminator, VGGFeatureExtractor, ImageEnhancer
from src.transformers import normalize_img_size, downward_img_quality
from src.utils import ImageDatasetWithTransforms, shuffle_lists_in_same_order, interpolate_models, \
    uniformity_loss, calculate_psnr, calculate_ssim, perceptal_loss, ReconstructionLoss
from PIL import Image
import torchvision.utils as vutils

import torch.nn.functional as F

nums_epoch = 50
warmUp_epochs = nums_epoch // 5


def train_example(rank, world_size, num_epochs, continue_training, prefix):
    # 初始化进程组
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'  # 选择一个未被占用的端口
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # 设置 GPU
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # 确保结果保存目录存在
    os.makedirs(f"results", exist_ok=True)

    lr_generator = 1e-4
    lr_dicriminator = lr_generator / 2

    g_criterion = ReconstructionLoss().to(device)

    generator = nn.parallel.DistributedDataParallel(SRResNet().to(device), device_ids=[rank])

    discriminator = nn.parallel.DistributedDataParallel(Discriminator().to(device), device_ids=[rank])

    vgg_extractor = VGGFeatureExtractor(layers=('conv3_3', 'conv4_3')).to(device)

    if continue_training:
        generator.load_state_dict(torch.load(os.path.join(os.getcwd(), 'results', f'{prefix}_generator_model_0.pth'),
                                             weights_only=True))
        discriminator.load_state_dict(
            torch.load(os.path.join(os.getcwd(), 'results', f'{prefix}_discriminator_model_0.pth'),
                       weights_only=True))
        lr_generator = lr_generator / 5
        lr_dicriminator = lr_dicriminator / 5
        prefix = "Post-Training"

    g_optimizer = optim.Adam(generator.parameters(), lr=lr_generator)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr_dicriminator)

    cosineLR = optim.lr_scheduler.CosineAnnealingLR

    linearLR = optim.lr_scheduler.LinearLR

    """lr_scheduler = cosineLR(optimizer=d_optimizer, T_max=num_epochs - warmUp_epochs, eta_min=lr_dicriminator/2)
    d_lr_scheduler = cosineLR(optimizer=d_optimizer, T_max=num_epochs - warmUp_epochs, eta_min=lr_dicriminator/2)"""
    lr_scheduler = linearLR(optimizer=g_optimizer, start_factor=1, end_factor=0.01, total_iters=num_epochs)
    d_lr_scheduler = linearLR(optimizer=d_optimizer, start_factor=1, end_factor=0.01, total_iters=num_epochs)

    # Define paths
    train_folder_path = os.path.join(os.getcwd(), 'data', 'train')
    val_folder_path = os.path.join(os.getcwd(), 'data', 'val')

    # Load full dataset
    train_data = ImageDatasetWithTransforms(train_folder_path, normalize_img_size, downward_img_quality)
    val_subset = ImageDatasetWithTransforms(val_folder_path, normalize_img_size, downward_img_quality)

    # Define split sizes (e.g., 70% train, 30% validation)
    split_ratio = 0.7
    train_size = int(split_ratio * len(train_data))
    val_size = len(train_data) - train_size

    # Split dataset into two parts
    train_subset, _ = random_split(train_data, [train_size, val_size])

    # Create samplers
    train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_subset, num_replicas=world_size, rank=rank, shuffle=True)

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=12, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=12, sampler=val_sampler, num_workers=0)

    psnrs = []
    ssims = []
    idx = []

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)  # 保证不同 GPU 训练的数据不重复
        val_sampler.set_epoch(epoch)

        # if epoch > -1:
        #    g_criterion = PerceptualLoss(device=device)# 内存不够，以后再说
        train_one_epoch(generator, train_loader, g_optimizer, vgg_extractor
                        , g_criterion, device, epoch, num_epochs, discriminator, d_optimizer, prefix)

        lr_scheduler.step()

        # d_lr_scheduler.step()

        if (epoch + 1) % 5 == 0:
            validate(generator, val_loader, device, epoch, prefix, dist.get_rank())

        psnr, ssim = compute_score(generator, val_loader, device)
        psnrs.append(psnr / 30)
        ssims.append(ssim)
        idx.append(epoch + 1)

    # Save the generator model's state_dict
    torch.save(generator.state_dict(), os.path.join(f'results', f'{prefix}_generator_model_{dist.get_rank()}.pth'))
    torch.save(discriminator.state_dict(),
               os.path.join(f'results', f'{prefix}_discriminator_model_{dist.get_rank()}.pth'))
    # Plotting the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(idx, psnrs, marker='o', linestyle='-', color='b', label='PNSR/30')
    plt.plot(idx, ssims, marker='o', linestyle='--', color='r', label='SSIM')
    plt.title('Rating Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Rating Value')
    plt.legend()
    plt.grid(True)

    # Save the plot as an image file in the 'results' directory
    plt.savefig(os.path.join(f'results', f'{prefix}training_loss_curve_{dist.get_rank()}.png'))

    dist.destroy_process_group()  # 训练结束后销毁进程组


def train_one_epoch(generator, train_loader, g_optimizer, vgg_extractor
                    , g_criterion, device, epoch, num_epochs, discriminator, d_optimizer, prefix):
    description = prefix
    t = tqdm(train_loader, desc=f"[{epoch + 1}/{num_epochs}] {description}")
    sum_g_loss = 0
    sum_d_loss = 0
    sum_c_loss = 0
    sum_p_loss = 0
    sum_g_d_loss = 0
    for batch_idx, (hr_imgs, lr_imgs) in enumerate(t):
        hr_imgs = hr_imgs.to(device)
        lr_imgs = lr_imgs.to(device)

        # d_loss = train_discriminator(discriminator, generator, hr_imgs, lr_imgs, d_optimizer)

        g_loss, com_loss, p_loss, g_d_loss = train_generator(generator, discriminator, lr_imgs, hr_imgs, vgg_extractor,
                                                             g_criterion, g_optimizer)

        sum_g_loss += g_loss
        # sum_d_loss += d_loss
        sum_c_loss += com_loss
        sum_p_loss += p_loss
        sum_g_d_loss += g_d_loss

        t.set_postfix(g=sum_g_loss / (batch_idx + 1), d=sum_d_loss / (batch_idx + 1))

    avg_loss = sum_g_loss / len(t)

    print(f"Epoch [{epoch + 1}/{num_epochs}] {description} Loss: {avg_loss:.6f}")
    print(f"com_loss: {sum_c_loss / len(t)}, tv_loss: {sum_p_loss / len(t)}, g_d_loss: {sum_g_d_loss / len(t)}")
    return avg_loss


def train_generator(generator, discriminator, lr_imgs, hr_imgs, vgg_extractor,
                    g_criterion, g_optimizer):
    torch.autograd.set_detect_anomaly(True)
    # --- Train Generator ---
    generator.train()
    discriminator.eval()

    sr_images = generator(lr_imgs)

    # fake_preds = discriminator(sr_images)

    # with torch.no_grad():
    #     real_preds = discriminator(hr_imgs)

    com_loss, tv_loss = g_criterion(hr_imgs, sr_images)
    # g_d_loss = torch.mean(torch.tanh(real_preds - fake_preds))
    g_d_loss = torch.tensor(0)
    g_loss = com_loss + tv_loss#  + g_d_loss

    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    loss_item = g_loss.item()

    del g_loss
    torch.cuda.empty_cache()  # Free unused memory

    return loss_item, com_loss.item(), tv_loss.item(), g_d_loss.item()


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
    image_enhancer = ImageEnhancer()
    with torch.no_grad():
        # 从验证集中获取一个batch
        hr_imgs, lr_imgs = next(iter(val_loader))
        hr_imgs = hr_imgs.to(device)
        lr_imgs = lr_imgs.to(device)
        sr_imgs = model(lr_imgs)
        # sr_imgs = image_enhancer.forward(sr_imgs)
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
    image_enhancer = ImageEnhancer()
    sum_psnr = 0
    sum_ssim = 0
    t = tqdm(val_loader, desc=f"validating:")
    cnt = 0
    for batch_idx, (hr_imgs, lr_imgs) in enumerate(t):
        if cnt == 30:
            break
        psnr = 0
        ssim = 0
        cnt += 1
        # 从验证集中获取一个batch
        hr_imgs = hr_imgs.to(device)
        lr_imgs = lr_imgs.to(device)
        with torch.no_grad():
            sr_imgs = model(lr_imgs)

        # sr_imgs = image_enhancer.forward(sr_imgs)

        for i in range(hr_imgs.size(0)):
            psnr += calculate_psnr(sr_imgs[i], hr_imgs[i])
            ssim += calculate_ssim(sr_imgs[i], hr_imgs[i])

        psnr /= hr_imgs.size(0)
        ssim /= hr_imgs.size(0)
        sum_psnr += psnr
        sum_ssim += ssim
        t.set_postfix(psnr=sum_psnr / cnt, ssim=sum_ssim / cnt)

    return sum_psnr / cnt, sum_ssim / cnt


if __name__ == "__main__":
    continue_training = False
    prefix = "Training"

    world_size = torch.cuda.device_count()
    mp.spawn(train_example, args=(world_size, nums_epoch, continue_training, prefix), nprocs=world_size)
