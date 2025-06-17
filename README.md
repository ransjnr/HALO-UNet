# HALO-UNet: Hardware Aware Lightweight Optimized U-Net for Thyroid Nodule Segmentation

## Overview

This repository implements the complete HALO-UNet methodology for efficient thyroid nodule segmentation in ultrasound images. The framework combines:

1. **Dynamic Noise-Adaptive Preprocessing (DNAP)** - Advanced preprocessing pipeline
2. **HALO-UNet Architecture** - Lightweight U-Net with depthwise separable convolutions
3. **Ultra-Light Attention (ULA) Module** - Efficient attention mechanism with entropy-based pruning

## 🏗️ Architecture

### HALO-UNet Model
- **Encoder-Decoder Structure**: U-Net style with skip connections
- **Depthwise Separable Convolutions**: Reduces parameters while maintaining performance
- **Ultra-Light Attention**: Strategic attention placement between encoder and decoder
- **Hardware Optimized**: Designed for resource-constrained environments

### Dynamic Noise-Adaptive Preprocessing (DNAP)
1. **Autoencoder Denoising**: 16→64→16 filter architecture with MSE loss
2. **Bilateral Filtering**: Edge-preserving smoothing (σ_color=10, σ_space=10)
3. **CLAHE Enhancement**: Contrast Limited Adaptive Histogram Equalization (clip_limit=2.0, tile_grid=8×8)
4. **Normalization**: Pixel intensity normalization to [0,1] range

### Ultra-Light Attention (ULA) Module
- **Channel Attention**: Squeeze-and-excite mechanism for channel weighting
- **Spatial Attention**: Lightweight 3×3 convolution for spatial focus
- **Entropy-Based Pruning**: Removes low-entropy attention heads (threshold=0.1)
- **Computational Efficiency**: Minimal overhead while improving performance

## 📁 Project Structure

```
├── model.py                  # HALO-UNet, ULA, and Autoencoder implementations
├── preprocessing.py          # DNAP pipeline implementation
├── dataset.py               # Ultrasound dataset with DNAP integration
├── train.py                 # Training script for HALO-UNet
├── utils.py                 # Evaluation metrics and utilities
├── evaluate_methodology.py  # Comprehensive evaluation script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd halo-unet

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation

Organize your ultrasound images and masks:
```
data/
├── train_images/
├── train_masks/
├── val_images/
└── val_masks/
```

### Training

1. **Train the Denoising Autoencoder** (automatically done first):
```python
python train.py
```

This will:
- First train the denoising autoencoder for DNAP
- Then train the HALO-UNet model with full DNAP preprocessing

### Evaluation

Run comprehensive methodology evaluation:
```python
python evaluate_methodology.py
```

This will provide:
- Model complexity comparison (HALO-UNet vs Traditional U-Net)
- DNAP preprocessing demonstration
- ULA attention module testing
- Segmentation quality metrics

## 💡 Key Features

### Efficiency Improvements
- **Parameter Reduction**: Significant reduction compared to traditional U-Net
- **Inference Speed**: Faster inference through optimized architecture
- **Memory Efficiency**: Lower memory footprint for deployment

### Medical Image Specific
- **Ultrasound Optimized**: Designed specifically for ultrasound characteristics
- **Noise Handling**: Advanced denoising through DNAP pipeline
- **Clinical Metrics**: Comprehensive medical segmentation evaluation

### Attention Mechanism
- **Entropy-Based Pruning**: Removes redundant attention patterns
- **Lightweight Design**: Minimal computational overhead
- **Adaptive Focus**: Dynamic attention based on feature importance

## 📊 Performance Metrics

The implementation includes comprehensive medical image segmentation metrics:

- **Dice Score**: Overlap-based similarity measure
- **IoU (Intersection over Union)**: Jaccard index for segmentation quality
- **Sensitivity/Recall**: True positive rate for nodule detection
- **Specificity**: True negative rate for healthy tissue
- **Precision**: Positive predictive value
- **F1 Score**: Harmonic mean of precision and recall

## 🔧 Configuration

### DNAP Parameters
```python
DNAP_CONFIG = {
    "bilateral_d": 10,
    "bilateral_sigma_color": 10.0,
    "bilateral_sigma_space": 10.0,
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid_size": (8, 8)
}
```

### Model Parameters
```python
# HALO-UNet Configuration
model = HALO_UNet(
    in_channels=3,
    out_channels=1,
    features=[32, 64, 128, 256]
)

# ULA Configuration
ula = UltraLightAttention(
    in_channels=64,
    entropy_threshold=0.1
)
```

## 📈 Usage Examples

### Basic Inference
```python
from model import HALO_UNet
from preprocessing import DynamicNoiseAdaptivePreprocessing

# Initialize model and preprocessing
model = HALO_UNet(in_channels=3, out_channels=1)
dnap = DynamicNoiseAdaptivePreprocessing()

# Process image
processed_image = dnap.process_image(input_image)
prediction = model(processed_image)
```

### Custom Training
```python
from dataset import UltrasoundDataset

# Create dataset with DNAP
dataset = UltrasoundDataset(
    image_dir="path/to/images",
    mask_dir="path/to/masks",
    use_dnap=True,
    dnap_config=DNAP_CONFIG
)
```

## 🔬 Methodology Details

### Autoencoder Architecture
- **Input**: Noisy ultrasound images
- **Encoder**: 3 layers (16→64→16 filters)
- **Decoder**: Mirror architecture
- **Loss**: Mean Squared Error (MSE)
- **Training**: Gaussian noise augmentation

### HALO-UNet Architecture
- **Encoder**: Depthwise separable convolutions with max pooling
- **Bottleneck**: ULA module for attention-guided feature processing
- **Decoder**: Transposed convolutions with skip connections
- **Output**: Binary segmentation mask

### Loss Function
Combined Dice-BCE Loss for optimal segmentation:
```python
loss = BCE_loss + Dice_loss
```

## 📋 Requirements

- Python 3.7+
- PyTorch 1.9+
- OpenCV 4.5+
- NumPy 1.21+
- Albumentations 1.1+
- Additional dependencies in `requirements.txt`

## 🎯 Applications

- **Thyroid Nodule Segmentation**: Primary application
- **Medical Image Segmentation**: Adaptable to other medical imaging tasks
- **Resource-Constrained Deployment**: Mobile and edge devices
- **Real-time Clinical Applications**: Fast inference for clinical workflow

## 🔄 Future Enhancements

- [ ] Multi-scale attention mechanisms
- [ ] 3D volume segmentation support
- [ ] Transfer learning capabilities
- [ ] Model quantization for mobile deployment
- [ ] Integration with DICOM workflows

## 📝 Citation

If you use this implementation in your research, please cite:

```bibtex
@article{halo-unet-2024,
  title={HALO-UNet: Hardware Aware Lightweight Optimized U-Net for Thyroid Nodule Segmentation},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Contact

For questions or support, please open an issue or contact [your-email@example.com] 