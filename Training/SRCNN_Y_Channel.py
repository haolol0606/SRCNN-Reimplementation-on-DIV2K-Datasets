import os
import copy
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import h5py
import numpy as np
import glob
import PIL.Image as pil_image
import torchvision.transforms.functional as TF
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm
from utils import AverageMeter, calc_psnr, convert_rgb_to_y

class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
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

class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file
        with h5py.File(h5_file, 'r') as f:  # Get length safely
            self.length = len(f['lr'])
    
    def __getitem__(self, idx):
        # Open file once per worker
        if not hasattr(self, 'h5'):
            self.h5 = h5py.File(self.h5_file, 'r', swmr=True)
            self.lr = self.h5['lr']
            self.hr = self.h5['hr']
        lr = self.lr[idx]
        hr = self.hr[idx]
        return torch.from_numpy(lr).unsqueeze(0), torch.from_numpy(hr).unsqueeze(0)
    
    def __len__(self):
        return self.length

class ValDataset(Dataset):
    def __init__(self, h5_file):
        super(ValDataset, self).__init__()
        self.h5_file = h5_file
        with h5py.File(h5_file, 'r') as f:
            self.keys = sorted(f['lr'].keys(), key=lambda x: int(x))
    
    def __getitem__(self, idx):
        if not hasattr(self, 'h5'):
            self.h5 = h5py.File(self.h5_file, 'r', swmr=True)
            self.lr = self.h5['lr']
            self.hr = self.h5['hr']
        key = self.keys[idx]
        lr = self.lr[key][:]
        hr = self.hr[key][:]
        return torch.from_numpy(lr).unsqueeze(0), torch.from_numpy(hr).unsqueeze(0)
    
    def __len__(self):
        return len(self.keys)

def prepare_datasets(args):
    print("Loading images...")

    # Load and sort images
    train_lr_images = sorted(glob.glob('{}/*'.format(args.train_lr_dir)))
    train_hr_images = sorted(glob.glob('{}/*'.format(args.train_hr_dir)))
    eval_lr_images = sorted(glob.glob('{}/*'.format(args.val_lr_dir)))
    eval_hr_images = sorted(glob.glob('{}/*'.format(args.val_hr_dir)))

    # Ensure equal number of LR and HR images
    assert len(train_lr_images) == len(train_hr_images), "Mismatch between train LR and HR images!"
    assert len(eval_lr_images) == len(eval_hr_images), "Mismatch between eval LR and HR images!"

    # Prepare training dataset
    prepare_train_dataset(args, train_lr_images, train_hr_images)
    
    # Prepare evaluation dataset
    prepare_val_dataset(args, eval_lr_images, eval_hr_images)

def prepare_train_dataset(args, lr_images, hr_images):
    print("Preparing training dataset...")
    
    with h5py.File(args.train_file, 'w') as h5_file:
        lr_dataset = h5_file.create_dataset('lr', shape=(0, args.patch_size, args.patch_size), 
                                            maxshape=(None, args.patch_size, args.patch_size), 
                                            dtype=np.float32, chunks=(1, args.patch_size, args.patch_size))

        hr_dataset = h5_file.create_dataset('hr', shape=(0, args.patch_size, args.patch_size), 
                                            maxshape=(None, args.patch_size, args.patch_size), 
                                            dtype=np.float32, chunks=(1, args.patch_size, args.patch_size))

        with tqdm(total=len(lr_images), desc="Processing Train Images") as pbar:
            for lr_path, hr_path in zip(lr_images, hr_images):
                hr = pil_image.open(hr_path).convert('RGB')
                lr = pil_image.open(lr_path).convert('RGB')

                # Upscale LR to HR size
                lr = lr.resize((hr.width, hr.height), resample=pil_image.BICUBIC)

                hr = np.array(hr).astype(np.float32)
                lr = np.array(lr).astype(np.float32)
                hr = convert_rgb_to_y(hr)
                lr = convert_rgb_to_y(lr)

                lr_patches = []
                hr_patches = []

                for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
                    for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
                        lr_patches.append(lr[i:i + args.patch_size, j:j + args.patch_size])
                        hr_patches.append(hr[i:i + args.patch_size, j:j + args.patch_size])

                if len(lr_patches) == 0:
                    pbar.update(1)
                    continue  

                lr_patches = (np.array(lr_patches, dtype=np.float32) / 255.0).astype(np.float32)
                hr_patches = (np.array(hr_patches, dtype=np.float32) / 255.0).astype(np.float32)

                lr_dataset.resize(lr_dataset.shape[0] + lr_patches.shape[0], axis=0)
                hr_dataset.resize(hr_dataset.shape[0] + hr_patches.shape[0], axis=0)

                lr_dataset[-lr_patches.shape[0]:] = lr_patches
                hr_dataset[-hr_patches.shape[0]:] = hr_patches

                pbar.update(1)

    print("Training dataset created.")

def prepare_val_dataset(args, lr_images, hr_images):
    print("Preparing evaluation dataset...")
    h5_file = h5py.File(args.eval_file, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    with tqdm(total=len(lr_images), desc="Processing Eval Images") as pbar:
        for i, (lr_path, hr_path) in enumerate(zip(lr_images, hr_images)):
            hr = pil_image.open(hr_path).convert('RGB')
            lr = pil_image.open(lr_path).convert('RGB')

            # Upscale LR to HR size
            lr = lr.resize((hr.width, hr.height), resample=pil_image.BICUBIC)

            hr = np.array(hr).astype(np.float32)
            lr = np.array(lr).astype(np.float32)
            lr = (convert_rgb_to_y(lr) / 255.0).astype(np.float32)
            hr = (convert_rgb_to_y(hr) / 255.0).astype(np.float32)

            lr_group.create_dataset(str(i), data=lr)
            hr_group.create_dataset(str(i), data=hr)

            pbar.update(1)

    h5_file.close()
    print("Evaluation dataset created.")

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class Args:
    train_hr_dir = 'D:/10 Epoch/SRCNN New/Datasets/DIV2K_train_HR'
    train_lr_dir = 'D:/10 Epoch/SRCNN New/Datasets/DIV2K_train_LR_bicubic/X3_resized'
    val_hr_dir ='D:/10 Epoch/SRCNN New/Datasets/DIV2K_valid_HR'
    val_lr_dir = 'D:/10 Epoch/SRCNN New/Datasets/DIV2K_valid_LR_bicubic/X3_resized'
    train_file = 'D:/10 Epoch/SRCNN New/Outputs/train.h5'
    eval_file = 'D:/10 Epoch/SRCNN New/Outputs/eval.h5'
    outputs_dir = 'D:/10 Epoch/SRCNN New/Outputs/output'
    scale = 3
    patch_size = 33
    stride = 14
    lr = 1e-4
    batch_size = 256
    num_epochs = 400
    num_workers = 6
    seed = 123

if __name__ == '__main__':
    args = Args()
    
    prepare_datasets(args)

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    cudnn.deterministic = False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    model = SRCNN().to(device)
    model = model.to(memory_format=torch.channels_last)
    model.apply(weights_init)
    criterion = nn.MSELoss()
    optimizer = optim.SGD([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)

    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True,
                                  prefetch_factor=12)
    eval_dataset = ValDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    writer = SummaryWriter(log_dir=args.outputs_dir)
    scaler = torch.amp.GradScaler()
    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('Epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs = inputs.to(memory_format=torch.channels_last)
                labels = labels.to(memory_format=torch.channels_last)

                with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
                    preds = model(inputs)
                    loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))

        # Log training loss to TensorBoard
        writer.add_scalar('Loss/train', epoch_losses.avg, epoch)

        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        # === Validation Phase (Calculate Validation Loss & PSNR) ===
        model.eval()
        epoch_psnr = AverageMeter()
        epoch_val_loss = AverageMeter()
        image_list = []
        num_images_to_log = 5

        for idx, data in enumerate(eval_dataloader):
            if idx >= num_images_to_log:
                break

            inputs, labels = data  # inputs = LR image, labels = HR image
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                preds = model(inputs)
                preds = torch.clamp(preds, 0.0, 1.0)
                val_loss = criterion(preds, labels)
                epoch_val_loss.update(val_loss.item(), len(inputs))

            print("Min pixel value:", preds.min().item())
            print("Max pixel value:", preds.max().item())

            epoch_psnr.update(calc_psnr(preds, labels, max_val=1.0), len(inputs))

            # Select the first image from the batch
            lr_img = inputs[0].float()
            sr_img = preds[0].float()
            hr_img = labels[0].float()

            # Convert grayscale (1-channel) to 3-channel for visualization
            lr_img = lr_img.expand(3, -1, -1)
            sr_img = sr_img.expand(3, -1, -1)
            hr_img = hr_img.expand(3, -1, -1)

            # Define a fixed size (H, W) for all images
            fixed_size = (512, 512)

            # Resize LR, SR, and HR images to 512x512
            lr_img = TF.resize(lr_img, fixed_size)
            sr_img = TF.resize(sr_img, fixed_size)
            hr_img = TF.resize(hr_img, fixed_size)

            # Stack LR, SR, HR images horizontally: (C, H, 3*W)
            image_list.append(torch.cat([lr_img, sr_img, hr_img], dim=2))

        # Stack all collected images vertically to form a grid (5 rows, 1 column)
        if image_list:
            grid = make_grid(image_list, nrow=1, padding=2, normalize=True)  # 1 row, multiple images stacked vertically
            writer.add_image("Comparison/LR_SR_HR", grid, epoch)

        print('eval psnr: {:.4f}'.format(epoch_psnr.avg))
        print('validation loss: {:.4f}'.format(epoch_val_loss.avg))

        # Log PSNR to TensorBoard
        writer.add_scalar('PSNR/eval', epoch_psnr.avg, epoch)
        writer.add_scalar('Loss/val', epoch_val_loss.avg, epoch)

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    writer.close()
    print('best epoch: {}, psnr: {:.4f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))