import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from preprocessing import DynamicNoiseAdaptivePreprocessing

class UltrasoundDataset(Dataset):
    """
    Dataset for ultrasound thyroid nodule segmentation with DNAP preprocessing.
    """
    def __init__(self, image_dir, mask_dir, transform=None, use_dnap=True, dnap_config=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.use_dnap = use_dnap
        self.images = os.listdir(image_dir)
        
        # Initialize DNAP pipeline if requested
        if self.use_dnap:
            dnap_config = dnap_config or {}
            self.dnap = DynamicNoiseAdaptivePreprocessing(**dnap_config)
        else:
            self.dnap = None
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        
        # Load image and convert to RGB
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0
        
        # Apply DNAP preprocessing if enabled
        if self.use_dnap and self.dnap is not None:
            # Convert to float32 in [0, 1] range
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            
            # Apply DNAP pipeline
            image = self.dnap.process_image(image)
            
            # Convert back to uint8 for albumentations if needed
            if self.transform is not None:
                image = (image * 255).astype(np.uint8)
        
        # Apply transformations if provided
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        return image, mask

# Legacy dataset for compatibility
class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            
        return image, mask
        

        
