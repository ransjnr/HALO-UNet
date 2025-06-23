import torch
import torchvision
from dataset import CarvanaDataset, UltrasoundDataset
from torch.utils.data import DataLoader
import numpy as np

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
            num_workers=0,  # Set to 0 to avoid CUDA multiprocessing issues
    pin_memory=True,
):
    """Legacy function for compatibility."""
    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )
    
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )   
    
    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )   
    
    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )   
    
    return train_loader, val_loader

def calculate_metrics(predictions, targets, threshold=0.5):
    """
    Calculate comprehensive medical image segmentation metrics.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        targets: Ground truth masks
        threshold: Threshold for binary prediction
        
    Returns:
        Dictionary containing various metrics
    """
    # Apply sigmoid if logits are provided
    if predictions.max() > 1 or predictions.min() < 0:
        predictions = torch.sigmoid(predictions)
    
    # Convert to binary predictions
    preds_binary = (predictions > threshold).float()
    
    # Flatten tensors
    preds_flat = preds_binary.view(-1)
    targets_flat = targets.view(-1)
    
    # Basic counts
    tp = (preds_flat * targets_flat).sum().float()
    fp = (preds_flat * (1 - targets_flat)).sum().float()
    fn = ((1 - preds_flat) * targets_flat).sum().float()
    tn = ((1 - preds_flat) * (1 - targets_flat)).sum().float()
    
    # Calculate metrics
    dice_score = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)  # Recall/True Positive Rate
    specificity = tn / (tn + fp + 1e-8)  # True Negative Rate
    precision = tp / (tp + fp + 1e-8)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    
    # F1 score
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-8)
    
    return {
        'dice_score': dice_score.item(),
        'iou': iou.item(),
        'sensitivity': sensitivity.item(),
        'specificity': specificity.item(),
        'precision': precision.item(),
        'accuracy': accuracy.item(),
        'f1_score': f1_score.item(),
        'tp': tp.item(),
        'fp': fp.item(),
        'fn': fn.item(),
        'tn': tn.item()
    }

def check_accuracy(loader, model, device="cuda"):
    """Enhanced accuracy checking with medical image metrics."""
    model.eval()
    
    # Initialize metric accumulators
    total_metrics = {
        'dice_score': 0.0,
        'iou': 0.0,
        'sensitivity': 0.0,
        'specificity': 0.0,
        'precision': 0.0,
        'accuracy': 0.0,
        'f1_score': 0.0
    }
    
    num_batches = 0
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1) if len(y.shape) == 3 else y.to(device)
            
            preds = model(x)
            
            # Calculate metrics for this batch
            batch_metrics = calculate_metrics(preds, y)
            
            # Accumulate metrics
            for key in total_metrics.keys():
                total_metrics[key] += batch_metrics[key]
            
            num_batches += 1
    
    # Average metrics across all batches
    avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
    
    # Print results
    print(f"Validation Results:")
    print(f"Dice Score: {avg_metrics['dice_score']:.4f}")
    print(f"IoU: {avg_metrics['iou']:.4f}")
    print(f"Sensitivity (Recall): {avg_metrics['sensitivity']:.4f}")
    print(f"Specificity: {avg_metrics['specificity']:.4f}")
    print(f"Precision: {avg_metrics['precision']:.4f}")
    print(f"Accuracy: {avg_metrics['accuracy']:.4f}")
    print(f"F1 Score: {avg_metrics['f1_score']:.4f}")
    
    model.train()
    return avg_metrics

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    """Save predictions with better visualization for medical images."""
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            
        # Save predictions
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        
        # Save ground truth
        y_unsqueezed = y.unsqueeze(1) if len(y.shape) == 3 else y
        torchvision.utils.save_image(y_unsqueezed, f"{folder}/target_{idx}.png")
        
        # Save original images
        torchvision.utils.save_image(x, f"{folder}/input_{idx}.png")
        
        # Create overlay visualization
        try:
            # Normalize input for better visualization
            x_norm = (x - x.min()) / (x.max() - x.min())
            
            # Create colored overlay (prediction in red, ground truth in green)
            overlay = x_norm.clone()
            if overlay.shape[1] >= 3:  # RGB image
                overlay[:, 0, :, :] = torch.where(preds.squeeze(1) > 0.5, 1.0, overlay[:, 0, :, :])  # Red for predictions
                overlay[:, 1, :, :] = torch.where(y_unsqueezed.squeeze(1) > 0.5, 1.0, overlay[:, 1, :, :])  # Green for ground truth
            
            torchvision.utils.save_image(overlay, f"{folder}/overlay_{idx}.png")
        except:
            pass  # Skip overlay if there's an issue
        
        # Only save first few batches to avoid too many images
        if idx >= 5:
            break
        
    model.train()

def calculate_model_parameters(model):
    """Calculate and print model parameters and FLOPs estimation."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    return total_params, trainable_params

def test_model_inference_speed(model, input_shape=(1, 3, 256, 256), device="cuda", num_runs=100):
    """Test model inference speed."""
    model.eval()
    model.to(device)
    
    # Warm up
    dummy_input = torch.randn(input_shape).to(device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Time inference
    torch.cuda.synchronize() if device == "cuda" else None
    import time
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if device == "cuda" else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    fps = 1.0 / avg_time
    
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")
    
    return avg_time, fps

