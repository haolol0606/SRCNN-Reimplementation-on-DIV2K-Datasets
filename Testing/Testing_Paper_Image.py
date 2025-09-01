import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
import PIL.Image as pil_image
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from utils import calc_psnr, convert_rgb_to_y

# Define the SRCNN Model
class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# Load trained SRCNN model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SRCNN().to(device)
model.load_state_dict(torch.load("D:/10 Epoch/SRCNN New/Output2/output/x3/epoch_4856.pth", map_location=device))
model.eval()

def load_hr_image(image_path):
    img = pil_image.open(image_path).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img.to(device)

def downscale_image(hr_image, scale_factor=3):
    _, _, h, w = hr_image.shape
    transform = transforms.Resize([h // scale_factor, w // scale_factor], interpolation=InterpolationMode.BICUBIC)
    return transform(hr_image)

def upscale_image(lr_image, target_size):
    return TF.resize(lr_image, target_size, interpolation=InterpolationMode.BICUBIC)

def save_image(tensor, output_path):
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    pil_image.fromarray(img).save(output_path)

def extract_patch(image_tensor, position='left-bottom', size=50, scale=3):
    _, h, w = image_tensor.shape[-3:]
    x = 0 if position == 'left-bottom' else w - size
    y = h - size
    patch = image_tensor[..., y:y+size, x:x+size]
    enlarged_patch = TF.resize(patch, [size * scale, size * scale], interpolation=InterpolationMode.BICUBIC)
    return patch, enlarged_patch

def show_images_combined(image_data):
    num_images = len(image_data)
    fig, axes = plt.subplots(num_images, 6, figsize=(16, 10))
    plt.subplots_adjust(wspace=0.5, hspace=1.0) 

    for i, data in enumerate(image_data):
        lr_np, sr_np, hr_np, lr_patch_np, sr_patch_np, hr_patch_np, lr_psnr, sr_psnr = data
        titles = [
            f"Low-Resolution\nPSNR: {lr_psnr:.2f} dB",
            f"Super-Resolved\nPSNR: {sr_psnr:.2f} dB",
            "Ground Truth (HR)",
            "LR Patch (Enlarged)",
            "SR Patch (Enlarged)",
            "HR Patch (Enlarged)"
        ]
        images = [lr_np, sr_np, hr_np, lr_patch_np, sr_patch_np, hr_patch_np]

        for j in range(6):
            ax = axes[i, j] if num_images > 1 else axes[j]
            ax.imshow(np.clip(images[j], 0, 1))
            ax.set_title(titles[j], fontsize=10, pad=10) 
            ax.axis("off")

    plt.tight_layout()
    plt.show(block=True)

hr_paths = [
    "D:/10 Epoch/SRCNN_train/SRCNN/Test/Set5/butterfly_GT.bmp",
    "D:/10 Epoch/SRCNN_train/SRCNN/Test/Set14/ppt3.bmp",
    "D:/10 Epoch/SRCNN_train/SRCNN/Test/Set14/zebra.bmp",
]

output_dir = "D:/10 Epoch/SRCNN New/Outputs/output/x3/Image"
os.makedirs(output_dir, exist_ok=True)

scale_factor = 3
image_data = []

for i, hr_path in enumerate(hr_paths):
    hr_img = load_hr_image(hr_path)
    lr_img = downscale_image(hr_img, scale_factor)
    lr_upscaled = upscale_image(lr_img, hr_img.shape[2:])

    with torch.no_grad():
        sr_img = model(lr_upscaled)
        sr_img = torch.clamp(sr_img, 0, 1)
    
    sr_y = convert_rgb_to_y(sr_img)
    hr_y = convert_rgb_to_y(hr_img)
    lr_y = convert_rgb_to_y(lr_upscaled)

    sr_psnr = calc_psnr(sr_y, hr_y, max_val=1.0)
    lr_psnr = calc_psnr(lr_y, hr_y, max_val=1.0)
    
    save_image(lr_upscaled, os.path.join(output_dir, f"LR_Upscaled_{i+1}.png"))
    save_image(sr_img, os.path.join(output_dir, f"SR_{i+1}.png"))

    lr_patch, lr_patch_enlarged = extract_patch(lr_upscaled.squeeze(0), 'right-bottom')
    sr_patch, sr_patch_enlarged = extract_patch(sr_img.squeeze(0), 'right-bottom')
    hr_patch, hr_patch_enlarged = extract_patch(hr_img.squeeze(0), 'right-bottom')
    
    lr_np = lr_upscaled.squeeze(0).permute(1, 2, 0).cpu().numpy()
    sr_np = sr_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    hr_np = hr_img.squeeze(0).permute(1, 2, 0).cpu().numpy()

    image_data.append((lr_np, sr_np, hr_np, lr_patch_enlarged.permute(1, 2, 0).cpu().numpy(), 
                        sr_patch_enlarged.permute(1, 2, 0).cpu().numpy(),
                        hr_patch_enlarged.permute(1, 2, 0).cpu().numpy(),
                        lr_psnr, sr_psnr))

show_images_combined(image_data)
