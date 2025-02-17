import os
import random

import torch.multiprocessing as mp
import torch.distributed as dist

import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import SRResNet, ImageFingerPrint, PerceptualLoss
from src.transformers import normalize_img_size, downward_img_quality
from src.utils import ImageDatasetWithTransforms, shuffle_lists_in_same_order, interpolate_models, \
    uniformity_loss
from PIL import Image
import torchvision.utils as vutils

import torch.nn.functional as F

nums_model = 1  # 生成模型池大小
nums_epoch = 100


def train_example(rank, world_size, num_epochs, num_models):
    # 初始化进程组
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'  # 选择一个未被占用的端口
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # 设置 GPU
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # 确保结果保存目录存在
    os.makedirs(f"results{num_models}", exist_ok=True)

    lr_generator = 1e-4
    lr_image_finger_print = 1e-4

    g_criterion = torch.nn.L1Loss()

    image_finger_print = ImageFingerPrint().to(device)
    image_finger_print = nn.SyncBatchNorm.convert_sync_batchnorm(image_finger_print)
    image_finger_print = nn.parallel.DistributedDataParallel(image_finger_print, device_ids=[rank])

    model = [nn.parallel.DistributedDataParallel(SRResNet().to(device), device_ids=[rank])
             for _ in range(num_models)]

    optimizers = [optim.Adam(generator.parameters(), lr=lr_generator + random.uniform(-1e-5, 1e-5)) for generator in
                  model]
    d_optimizer = optim.Adam(image_finger_print.parameters(), lr=lr_image_finger_print)

    scheduler = optim.lr_scheduler.StepLR

    d_lr_scheduler = scheduler(
        optimizer=d_optimizer,
        step_size=5,
        gamma=0.65
    )
    g_lr_schedulers = []
    for optimizer in optimizers:
        lr_scheduler = scheduler(
            optimizer=optimizer,
            step_size=5,
            gamma=0.65
        )
        g_lr_schedulers.append(lr_scheduler)

    image_folder_path = os.path.join(os.getcwd(), 'data', 'train')
    train_data = ImageDatasetWithTransforms(image_folder_path, normalize_img_size, downward_img_quality)
    sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, sampler=sampler, num_workers=0)

    if dist.get_rank() == 0:
        image_folder_path = os.path.join(os.getcwd(), 'data', 'val')
        val_data = ImageDatasetWithTransforms(image_folder_path, normalize_img_size, downward_img_quality)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=8, num_workers=0)

    avg_losses = []

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # 保证不同 GPU 训练的数据不重复

        # if epoch > -1:
        #    g_criterion = PerceptualLoss(device=device)# 内存不够，以后再说
        gen_losses = [0 for i in range(len(model))]
        avg_loss = train_one_epoch(model, image_finger_print, train_loader, optimizers, d_optimizer
                                   , g_criterion, device, epoch, num_epochs, gen_losses)
        avg_losses.append(avg_loss)

        d_lr_scheduler.step()

        for lr_scheduler in g_lr_schedulers[1:]:
            lr_scheduler.step()

        # 将模型按照对比损失，从小到大排列
        shuffle_lists_in_same_order(model, optimizers, gen_losses)

        # 验证：每个epoch结束后随机取一个batch验证效果
        if (epoch + 1) % 5 == 0 and dist.get_rank() == 0:
            validate(model[-1], val_loader, device, epoch, num_models, "fingerprint")

    dist.destroy_process_group()  # 训练结束后销毁进程组

    # Save the generator model's state_dict
    # avg_losses[0] = 1 # 防止第一个损失太大带来的曲线偏离，无法看清后续的变化趋势
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


def train_one_epoch(model, image_finger_print, train_loader, g_optimizer, d_optimizer
                    , g_criterion, device, epoch, num_epochs, gen_losses):
    total_loss = 0
    description = "Training"
    t = tqdm(train_loader, desc=f"[{epoch + 1}/{num_epochs}] {description}")
    for batch_idx, (hr_imgs, lr_imgs) in enumerate(t):

        hr_imgs = hr_imgs.to(device)
        lr_imgs = lr_imgs.to(device)

        better_model = model[-1]
        g_loss = 0

        d_loss = train_image_finger_print(image_finger_print, hr_imgs, d_optimizer)

        for i in range(len(model)):
            generator = model[i]
            optimizer = g_optimizer[i]

            g_loss = train_generator(generator, image_finger_print, lr_imgs, hr_imgs,
                                     g_criterion, optimizer,
                                     i, better_model, gen_losses)

            gen_losses[i] += g_loss

        t.set_postfix(g=g_loss, d=d_loss)
        total_loss += g_loss

    avg_loss = total_loss / len(train_loader)

    for i in range(len(gen_losses)):
        gen_losses[i] /= len(train_loader)  # 直接修改原列表

    print(f"Epoch [{epoch + 1}/{num_epochs}] {description} Loss: {avg_loss:.6f}")
    return avg_loss


def train_generator(generator, image_finger_print, lr_imgs, hr_imgs,
                    g_criterion, g_optimizer,
                    model_idx, better_model, gen_losses):
    torch.autograd.set_detect_anomaly(True)
    # --- Train Generator ---
    generator.train()

    sr_images = generator(lr_imgs)

    fake_preds = image_finger_print(sr_images)
    real_preds = image_finger_print(hr_imgs)

    g_loss = g_criterion(fake_preds, real_preds)  # 高维比对

    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    generator.eval()

    loss_item = g_loss.item()

    del g_loss
    torch.cuda.empty_cache()  # Free unused memory

    return loss_item


def train_image_finger_print(image_finger_print, hr_imgs, d_optimizer):
    torch.autograd.set_detect_anomaly(True)
    # --- Train image_finger_print ---
    image_finger_print.train()

    # Get image_finger_print predictions
    real_preds = image_finger_print(hr_imgs)

    d_loss = uniformity_loss(real_preds)

    # Update image_finger_print
    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    image_finger_print.eval()

    loss_item = d_loss.item()

    del d_loss  # Delete loss tensor
    torch.cuda.empty_cache()  # Free unused memory

    return loss_item


def validate(model, val_loader, device, epoch, num_models, desc):
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
        save_path = os.path.join(f"results{num_models}", f"{desc}_epoch_{epoch + 1}_comparison.png")
        vutils.save_image(comparison_grid, save_path)
        print(f"Epoch {epoch + 1}: Comparison image saved to {save_path}")
    return save_path


if __name__ == "__main__":
    print(f'Training with {nums_model} generators competing!')

    world_size = torch.cuda.device_count()
    mp.spawn(train_example, args=(world_size, nums_epoch, nums_model), nprocs=world_size)
