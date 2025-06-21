import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import glob
from facenet_pytorch import InceptionResnetV1
os.makedirs('generated_images', exist_ok=True)
class FaceDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_paths = glob.glob(os.path.join(img_dir, '*.jpg')) + \
                         glob.glob(os.path.join(img_dir, '*.png'))
        self.transform = transform
        if not self.img_paths:
            raise FileNotFoundError(f"错误：在 '{img_dir}' 中未找到任何 .jpg 或 .png 格式的图片。")
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path
    
class Generator(nn.Module):
    def __init__(self, noise_dim=100, img_channels=3, feature_g=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            self._block(noise_dim, feature_g * 8, 4, 1, 0),
            self._block(feature_g * 8, feature_g * 4, 4, 2, 1),
            self._block(feature_g * 4, feature_g * 2, 4, 2, 1),
            self._block(feature_g * 2, feature_g, 4, 2, 1),
            nn.ConvTranspose2d(feature_g, img_channels, 4, 2, 1),
            nn.Tanh()
        )
    def _block(self, in_channels, out_channels, k, s, p):
        return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, k, s, p, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))
    def forward(self, input):
        return self.main(input)
class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feature_d=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(img_channels, feature_d, 4, 2, 1), nn.LeakyReLU(0.2),
            self._block(feature_d, feature_d * 2, 4, 2, 1),
            self._block(feature_d * 2, feature_d * 4, 4, 2, 1),
            self._block(feature_d * 4, feature_d * 8, 4, 2, 1),
            nn.Conv2d(feature_d * 8, 1, 4, 1, 0), nn.Sigmoid()
        )
    def _block(self, in_channels, out_channels, k, s, p):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, k, s, p, bias=False), nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True))
    def forward(self, input):
        return self.main(input).view(-1)
    
def apply_mask(image_tensor, mask_size=4):
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    _, _, h, w = image_tensor.shape
    top = np.random.randint(0, h - mask_size)
    left = np.random.randint(0, w - mask_size)
    masked_image = image_tensor.clone()
    masked_image[:, :, top:top+mask_size, left:left+mask_size] = -1
    return masked_image
def save_tensor_as_image(tensor, filename):
    tensor = tensor.detach().cpu().squeeze(0)
    image = transforms.ToPILImage()((tensor + 1) / 2)
    image.save(filename)
def preprocess_for_facenet(tensor_img):
    tensor_img_resized = F.resize(tensor_img, (160, 160), antialias=True)
    return tensor_img_resized


epochs = 2000
lr = 0.0002
beta1 = 0.5
batch_size = 16
noise_dim = 100
image_size = 64
data_dir = './data/face_subset/n000014'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"device:{device}")

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
try:
    dataset = FaceDataset(img_dir=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    _, target_identity_path = dataset[0]
except FileNotFoundError as e:
    exit()
# print("加载 InceptionResnetV1 ")
facenet_model = InceptionResnetV1(pretrained='vggface2').to(device)
facenet_model.eval()
for param in facenet_model.parameters():
    param.requires_grad = False
# print("123456")
target_img_pil = Image.open(target_identity_path).convert('RGB')
target_img_tensor = transform(target_img_pil).unsqueeze(0).to(device)
with torch.no_grad():
    target_img_preprocessed = preprocess_for_facenet(target_img_tensor)
    target_embedding = facenet_model(target_img_preprocessed).detach()

netG = Generator(noise_dim=noise_dim).to(device)
netD = Discriminator().to(device)
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
adversarial_loss = nn.BCELoss()
identity_loss_fn = nn.MSELoss()
print("开始训练...")
total_epochs = epochs
phase1_epochs = total_epochs // 2
for epoch in range(total_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        current_batch_size = real_images.size(0)
        noise = torch.randn(current_batch_size, noise_dim, 1, 1, device=device)
        fake_images = netG(noise)
        if epoch < phase1_epochs:
            # if epoch == 0 and i == 0:
            #     print("awa")
            netD.zero_grad()
            real_labels = torch.full((current_batch_size,), 1.0, device=device)
            fake_labels = torch.full((current_batch_size,), 0.0, device=device)
            output_real = netD(real_images)
            lossD_real = adversarial_loss(output_real, real_labels)
            output_fake = netD(fake_images.detach())
            lossD_fake = adversarial_loss(output_fake, fake_labels)
            lossD = (lossD_real + lossD_fake) / 2
            lossD.backward()
            optimizerD.step()
            netG.zero_grad()
            output_g = netD(fake_images)
            lossG_adv = adversarial_loss(output_g, real_labels)
            fake_images_preprocessed = preprocess_for_facenet(fake_images)
            fake_embeddings = facenet_model(fake_images_preprocessed)
            lossG_identity = identity_loss_fn(fake_embeddings, target_embedding.repeat(current_batch_size, 1))
            lambda_identity = 0.5
            lossG_total = lossG_adv + lambda_identity * lossG_identity
            lossG_total.backward()
            optimizerG.step()
            if i % 10 == 0:
                 print(f"[{i}/{len(dataloader)}] Loss_D: {lossD.item():.4f}, Loss_G_adv: {lossG_adv.item():.4f}, Loss_G_identity: {lossG_identity.item():.4f}")
        else:
            # if epoch == phase1_epochs and i == 0:
                # print("bwb")
            netG.zero_grad()
            masked_images = torch.cat([apply_mask(img) for img in fake_images])
            masked_images_preprocessed = preprocess_for_facenet(masked_images)
            masked_embeddings = facenet_model(masked_images_preprocessed)
            lossG = identity_loss_fn(masked_embeddings, target_embedding.repeat(current_batch_size, 1))
            lossG.backward()
            optimizerG.step()
            if i % 10 == 0:
                print(f"[{i}/{len(dataloader)}] Loss_G {lossG.item():.4f}")
    with torch.no_grad():
        fixed_noise = torch.randn(1, noise_dim, 1, 1, device=device)
        final_image = netG(fixed_noise).detach().cpu()
        save_tensor_as_image(final_image, f"generated_images/final_epoch_{epoch+1}.jpg")