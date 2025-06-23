import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Union, Tuple
from model import DenoisingAutoencoder

class DynamicNoiseAdaptivePreprocessing:
    """
    Dynamic Noise-Adaptive Preprocessing (DNAP) pipeline for ultrasound images.
    
    Pipeline steps:
    1. Autoencoder-based denoising (16→64→16 filters, MSE loss)
    2. Bilateral filtering (σ_color=10, σ_space=10)
    3. CLAHE (clip_limit=2.0, tile_grid=8×8)
    4. Normalization to [0,1] range
    """
    
    def __init__(self, 
                 device: str = "cuda",
                 autoencoder_path: str = None,
                 bilateral_d: int = 10,
                 bilateral_sigma_color: float = 10.0,
                 bilateral_sigma_space: float = 10.0,
                 clahe_clip_limit: float = 2.0,
                 clahe_tile_grid_size: Tuple[int, int] = (8, 8)):
        """
        Initialize DNAP pipeline.
        
        Args:
            device: Device to run the autoencoder on
            autoencoder_path: Path to pretrained autoencoder weights
            bilateral_d: Diameter of each pixel neighborhood for bilateral filter
            bilateral_sigma_color: Filter sigma in the color space
            bilateral_sigma_space: Filter sigma in the coordinate space
            clahe_clip_limit: Threshold for contrast limiting
            clahe_tile_grid_size: Size of the neighborhood for CLAHE
        """
        self.device = device
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        
        # Initialize autoencoder
        self.autoencoder = DenoisingAutoencoder(in_channels=3)
        if autoencoder_path:
            self.autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=device))
        self.autoencoder.to(device)
        self.autoencoder.eval()
        
        # Initialize CLAHE
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size
        )
    
    def denoise_with_autoencoder(self, image: np.ndarray) -> np.ndarray:
        """
        Apply autoencoder-based denoising.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Denoised image as numpy array (H, W, C)
        """
        if self.autoencoder is None:
            return image
        
        # Convert to tensor and normalize
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        
        # Convert to tensor format (1, C, H, W) on CPU first to avoid multiprocessing issues
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
        
        # Move to device if available (only happens in main process now with num_workers=0)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            denoised_tensor = self.autoencoder(image_tensor)
        
        # Convert back to numpy (H, W, C)
        denoised = denoised_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        
        # Clip values to [0, 1] range
        denoised = np.clip(denoised, 0, 1)
        
        return denoised
    
    def apply_bilateral_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filtering for edge-preserving smoothing.
        
        Args:
            image: Input image as numpy array (H, W, C) in [0, 1] range
            
        Returns:
            Filtered image as numpy array (H, W, C) in [0, 1] range
        """
        # Convert to uint8 for OpenCV
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Apply bilateral filter
        filtered = cv2.bilateralFilter(
            image_uint8,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space
        )
        
        # Convert back to float32 in [0, 1] range
        filtered = filtered.astype(np.float32) / 255.0
        
        return filtered
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
        
        Args:
            image: Input image as numpy array (H, W, C) in [0, 1] range
            
        Returns:
            CLAHE-enhanced image as numpy array (H, W, C) in [0, 1] range
        """
        # Convert to uint8
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Convert to LAB color space for better results
        lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to the L channel
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Convert back to float32 in [0, 1] range
        enhanced = enhanced.astype(np.float32) / 255.0
        
        return enhanced
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize pixel intensity values to [0, 1] range.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Normalized image in [0, 1] range
        """
        # Get min and max values
        min_val = np.min(image)
        max_val = np.max(image)
        
        # Avoid division by zero
        if max_val - min_val == 0:
            return image
        
        # Normalize to [0, 1]
        normalized = (image - min_val) / (max_val - min_val)
        
        return normalized
    
    def process_image(self, image: Union[np.ndarray, str]) -> np.ndarray:
        """
        Apply the complete DNAP pipeline to an image.
        
        Args:
            image: Input image as numpy array (H, W, C) or path to image file
            
        Returns:
            Processed image as numpy array (H, W, C) in [0, 1] range
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ensure image is in the correct format
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Step 1: Autoencoder-based denoising
        denoised = self.denoise_with_autoencoder(image)
        
        # Step 2: Bilateral filtering
        filtered = self.apply_bilateral_filter(denoised)
        
        # Step 3: CLAHE enhancement
        enhanced = self.apply_clahe(filtered)
        
        # Step 4: Final normalization
        normalized = self.normalize_image(enhanced)
        
        return normalized
    
    def process_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Process a batch of images through the DNAP pipeline.
        
        Args:
            images: Batch of images as tensor (B, C, H, W)
            
        Returns:
            Processed batch as tensor (B, C, H, W)
        """
        batch_size = images.shape[0]
        processed_batch = []
        
        for i in range(batch_size):
            # Convert tensor to numpy (H, W, C)
            img_np = images[i].cpu().numpy().transpose(1, 2, 0)
            
            # Process image
            processed_np = self.process_image(img_np)
            
            # Convert back to tensor (C, H, W)
            processed_tensor = torch.from_numpy(processed_np.transpose(2, 0, 1))
            processed_batch.append(processed_tensor)
        
        # Stack to create batch
        processed_batch = torch.stack(processed_batch).to(images.device)
        
        return processed_batch

def train_denoising_autoencoder(
    dataloader,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    device: str = "cuda",
    save_path: str = "denoising_autoencoder.pth"
):
    """
    Train the denoising autoencoder.
    
    Args:
        dataloader: DataLoader for training data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on
        save_path: Path to save trained model
    """
    # Initialize model
    autoencoder = DenoisingAutoencoder(in_channels=3).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    
    autoencoder.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            
            # Add noise to create noisy images
            noise = torch.randn_like(images) * 0.1  # Gaussian noise
            noisy_images = torch.clamp(images + noise, 0, 1)
            
            # Forward pass
            optimizer.zero_grad()
            reconstructed = autoencoder(noisy_images)
            loss = criterion(reconstructed, images)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    
    # Save the trained model
    torch.save(autoencoder.state_dict(), save_path)
    print(f'Trained autoencoder saved to {save_path}')

# Example usage and testing
if __name__ == "__main__":
    # Initialize DNAP pipeline
    dnap = DynamicNoiseAdaptivePreprocessing()
    
    # Test with random image
    test_image = np.random.rand(256, 256, 3).astype(np.float32)
    processed = dnap.process_image(test_image)
    
    print(f"Original image shape: {test_image.shape}")
    print(f"Processed image shape: {processed.shape}")
    print(f"Original range: [{test_image.min():.3f}, {test_image.max():.3f}]")
    print(f"Processed range: [{processed.min():.3f}, {processed.max():.3f}]") 