import torch
from src.models import SRResNet
from src.utils import load_image, tensor_to_image
from src.train import train_example


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载预训练模型（假设预训练权重保存在 'srresnet.pth'）
    model = SRResNet().to(device)
    model.load_state_dict(torch.load("srresnet.pth", map_location=device))
    model.eval()

    # 加载并预处理低分辨率图像
    lr_tensor = load_image("low_resolution_image.jpg").to(device)

    # 推断生成超分辨率图像
    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    # 后处理并保存显示
    sr_image = tensor_to_image(sr_tensor)
    sr_image.save("output_sr.jpg")
    sr_image.show()


if __name__ == "__main__":
    train_example(20, 3)
