import os
from PIL import Image
import glob

# Directories
lr_dir = 'D:/10 Epoch/SRCNN New/Datasets/DIV2K_valid_LR_bicubic/X4'
output_dir = 'D:/10 Epoch/SRCNN New/Datasets/DIV2K_valid_LR_bicubic/X4_upscaled'
scale = 4  # Upscaling factor

# Create output directory if not exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process images
for lr_path in glob.glob(f"{lr_dir}/*"):
    lr_image = Image.open(lr_path).convert("RGB")
    
    # Compute new upscaled dimensions
    new_w, new_h = lr_image.width * scale, lr_image.height * scale
    
    # Upscale image using bicubic interpolation
    upscaled_lr = lr_image.resize((new_w, new_h), Image.BICUBIC)
    
    # Save the upscaled image
    save_path = os.path.join(output_dir, os.path.basename(lr_path))
    upscaled_lr.save(save_path)

print("Preprocessing complete: Upscaled LR images saved.")
