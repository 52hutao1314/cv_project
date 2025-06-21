import torch
import torchvision.utils as vutils
import argparse
from gan_models import Generator, NZ
parser = argparse.ArgumentParser(description="使用训练好的GAN生成器生成单张图片")
parser.add_argument('--weights', type=str, default='generator.pth', help='生成器权重文件路径')
parser.add_argument('--output', type=str, default='fake_image.png', help='输出图片的文件名')
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)
netG.load_state_dict(torch.load(args.weights, map_location=device))
netG.eval()
noise = torch.randn(1, NZ, 1, 1, device=device)
with torch.no_grad():
    fake_image = netG(noise).detach().cpu()
vutils.save_image(fake_image, args.output, normalize=True)
print(f"图像已成功生成并保存至: {args.output}")