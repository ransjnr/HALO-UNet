import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import HALO_UNet, DenoisingAutoencoder
from dataset import UltrasoundDataset
from preprocessing import train_denoising_autoencoder
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 50
NUM_WORKERS = 2
IMAGE_HEIGHT = 256  # Standard size for ultrasound images
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/trainval-image/"
TRAIN_MASK_DIR = "data/trainval-mask/"
VAL_IMG_DIR = "data/test-image/"
VAL_MASK_DIR = "data/test-mask/"

# DNAP Configuration
DNAP_CONFIG = {
    "device": DEVICE,
    "autoencoder_path": "denoising_autoencoder.pth",
    "bilateral_d": 10,
    "bilateral_sigma_color": 10.0,
    "bilateral_sigma_space": 10.0,
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid_size": (8, 8)
}

def train_halo_unet():
    """Train the HALO-UNet model for thyroid nodule segmentation."""
    
    # Define transforms
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    # Initialize HALO-UNet model
    model = HALO_UNet(in_channels=3, out_channels=1).to(DEVICE)
    
    # Use Dice Loss + BCE for better segmentation performance
    class DiceBCELoss(nn.Module):
        def __init__(self, weight=None, size_average=True):
            super(DiceBCELoss, self).__init__()

        def forward(self, inputs, targets, smooth=1):
            # BCE loss
            BCE = nn.BCEWithLogitsLoss()(inputs, targets)
            
            # Apply sigmoid to inputs for Dice calculation
            inputs = torch.sigmoid(inputs)
            
            # Flatten tensors
            inputs = inputs.view(-1)
            targets = targets.view(-1)
            
            # Dice coefficient
            intersection = (inputs * targets).sum()
            dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
            dice_loss = 1 - dice
            
            # Combined loss
            total_loss = BCE + dice_loss
            
            return total_loss

    loss_fn = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Load data loaders with DNAP preprocessing
    train_loader, val_loader = get_ultrasound_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
        use_dnap=True,
        dnap_config=DNAP_CONFIG,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        model.train()
        loop = tqdm(train_loader)

        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=DEVICE)
            targets = targets.float().unsqueeze(1).to(device=DEVICE)

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions, targets)

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update progress bar
            loop.set_postfix(loss=loss.item())

        # Save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # Check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # Save some examples
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )

def get_ultrasound_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
    use_dnap=True,
    dnap_config=None,
):
    """Get data loaders for ultrasound dataset with optional DNAP preprocessing."""
    
    train_ds = UltrasoundDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
        use_dnap=use_dnap,
        dnap_config=dnap_config,
    )
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )   
    
    val_ds = UltrasoundDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
        use_dnap=use_dnap,
        dnap_config=dnap_config,
    )   
    
    val_loader = torch.utils.data.DataLoader(
        dataset=val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )   
    
    return train_loader, val_loader

def train_autoencoder_first():
    """Train the denoising autoencoder first before main training."""
    print("Training denoising autoencoder...")
    
    # Simple transforms for autoencoder training
    autoencoder_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
    
    # Get data loader without DNAP (for training the autoencoder)
    train_loader, _ = get_ultrasound_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        autoencoder_transform,
        autoencoder_transform,
        NUM_WORKERS,
        PIN_MEMORY,
        use_dnap=False,  # Don't use DNAP when training the autoencoder
    )
    
    # Train the autoencoder
    train_denoising_autoencoder(
        dataloader=train_loader,
        num_epochs=50,
        learning_rate=1e-3,
        device=DEVICE,
        save_path="denoising_autoencoder.pth"
    )
    
    print("Autoencoder training completed!")

if __name__ == "__main__":
    # First train the denoising autoencoder
    train_autoencoder_first()
    
    # Then train the main HALO-UNet model
    train_halo_unet()
        












