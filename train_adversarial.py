# python train_adversarial.py --target_id n000014 --rep_path representations.pkl --epochs 2000 --id_loss_weight 2.0 --dataroot ./data/face_subset
# pip3 install torch torchvision torchaudio -f https://mirrors.aliyun.com/pytorch-wheels/cu126
# 导入所有必要的库
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm
import argparse
import os

# 从我们自己的文件中导入模型和工具函数
from gan_models import Generator, Discriminator, NZ
from recognition_utils import get_target_embedding, get_embedding_from_tensor

def main():
    parser = argparse.ArgumentParser(description="使用GAN生成对抗性人脸来欺骗人脸识别模型")
    parser.add_argument('--dataroot', type=str, required=True, help='包含真实人脸图片的根目录路径')
    parser.add_argument('--target_id', type=str, required=True, help='你想在特征库中模仿的目标人物ID (文件夹名)')
    parser.add_argument('--rep_path', type=str, default='my_representations.pkl', help='你的.pkl人脸特征库文件路径')
    parser.add_argument('--workers', type=int, help='DataLoader使用的工作进程数', default=2)
    parser.add_argument('--batch_size', type=int, default=16, help='训练中的批量大小')
    parser.add_argument('--image_size', type=int, default=64, help='生成的图片尺寸 (高度和宽度)')
    parser.add_argument('--epochs', type=int, default=100, help='总共的训练轮次')
    parser.add_argument('--lr', type=float, default=0.0002, help='Adam优化器的学习率')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam优化器的beta1参数')
    parser.add_argument('--id_loss_weight', type=float, default=1.0, help='身份损失相对于对抗损失的权重')
    parser.add_argument('--output_dir', type=str, default='generated_faces', help='保存生成图片的目录')
    args = parser.parse_args()
    print("脚本参数设置: ", args)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f"{device}")
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    dataset = dset.ImageFolder(root=args.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(args.image_size),
                                   transforms.CenterCrop(args.image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.workers)
    adversarial_criterion = nn.BCELoss()
    identity_criterion = nn.MSELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    target_embedding = get_target_embedding(args.rep_path, args.target_id).to(device)
    fixed_noise = torch.randn(64, NZ, 1, 1, device=device)
    real_label_val = 1.
    fake_label_val = 0.
    print("开始对抗训练...")
    for epoch in range(args.epochs):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"轮次 {epoch+1}/{args.epochs}")
        for i, data in progress_bar:
            netD.zero_grad()
            real_images = data[0].to(device)
            b_size = real_images.size(0)
            label = torch.full((b_size,), real_label_val, dtype=torch.float, device=device)
            output = netD(real_images).view(-1)
            errD_real = adversarial_criterion(output, label)
            errD_real.backward()
            noise = torch.randn(b_size, NZ, 1, 1, device=device)
            fake_images = netG(noise)
            label.fill_(fake_label_val)
            output = netD(fake_images.detach()).view(-1)
            errD_fake = adversarial_criterion(output, label)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()
            netG.zero_grad()
            label.fill_(real_label_val)
            output = netD(fake_images).view(-1)

            errG_adversarial = adversarial_criterion(output, label)
            fake_images_resized = nn.functional.interpolate(fake_images, size=(224, 224), mode='bilinear', align_corners=False)
            generated_embeddings = get_embedding_from_tensor(fake_images_resized, device)
            target_embedding_batch = target_embedding.unsqueeze(0).repeat(b_size, 1)
            errG_identity = identity_criterion(generated_embeddings, target_embedding_batch)
            errG_total = errG_adversarial + args.id_loss_weight * errG_identity
            errG_total.backward()
            optimizerG.step()
            
            progress_bar.set_postfix(Loss_D=f'{errD.item():.4f}', Loss_G=f'{errG_total.item():.4f}', Id_Loss=f'{errG_identity.item():.4f}')

        if (epoch % 100 == 0) or (epoch == args.epochs - 1):
            with torch.no_grad():
                fake_samples = netG(fixed_noise).detach().cpu()
            vutils.save_image(fake_samples,
                              f'{args.output_dir}/fake_samples_epoch_{epoch}.png',
                              normalize=True)

    print("训练完成。")
    generator_weights_path = 'generator.pth'
    torch.save(netG.state_dict(), generator_weights_path)

if __name__ == '__main__':
    main()
