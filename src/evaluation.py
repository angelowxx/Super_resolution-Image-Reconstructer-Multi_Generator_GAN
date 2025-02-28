import argparse
import os
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import SRResNet, ImageEnhancer
from src.transformers import add_noise, downward_img_quality
from src.utils import ImageDatasetWithTransforms, calculate_psnr, calculate_ssim, ImageDataset


def evaluate_model(dataset, lr_path, hr_path):
    # Define paths
    model_path = os.path.join(os.getcwd(), 'results', 'Training_generator_model_0.pth')
    eval_folder_path = dataset
    eval_data = ImageDataset(eval_folder_path, lr_path, hr_path)
    eval_loader = DataLoader(eval_data, batch_size=6)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SRResNet().to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # Remove the "module." prefix if present
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    image_enhancer = ImageEnhancer()

    description = "evaluating"
    t = tqdm(eval_loader, desc=f"{description}")
    t_psnr = 0
    t_ssim = 0
    for batch_idx, (hr_imgs, lr_imgs) in enumerate(t):
        hr_imgs = hr_imgs.to(device)
        lr_imgs = lr_imgs.to(device)

        sr_imgs = model(lr_imgs)
        sr_imgs = image_enhancer.forward(sr_imgs)

        psnr = calculate_psnr(sr_imgs, hr_imgs)
        ssim = calculate_ssim(sr_imgs, hr_imgs)

        t.set_postfix(psnr=psnr, ssim=ssim)
        t_psnr += psnr
        t_ssim += ssim
    print(f'average psnr = {t_psnr / len(t)}, average ssim = {t_ssim / len(t)}')


if __name__ == "__main__":
    print(f'evaluating!')
    cmdline_parser = argparse.ArgumentParser('evaluating sr')
    cmdline_parser.add_argument('-D', '--data_dir',
                                default=os.path.join(os.getcwd(), 'eval'),
                                help='where the evaluation dataset stored')
    cmdline_parser.add_argument('-lr', '--lr_dir',
                                default='LRbicx4',
                                help='where low resolution images stored under eval dataset')
    cmdline_parser.add_argument('-hr', '--hr_dir',
                                default='original',
                                help='where high resolution images stored under eval dataset')

    args, unknowns = cmdline_parser.parse_known_args()

    evaluate_model(dataset=args.data_dir, lr_path=args.lr_dir, hr_path=args.hr_dir)
