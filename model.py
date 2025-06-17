import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import math

class DenoisingAutoencoder(nn.Module):
    """
    Shallow convolutional autoencoder for denoising ultrasound images.
    Architecture: 16→64→16 filters with MSE loss.
    """
    def __init__(self, in_channels=3):
        super(DenoisingAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Output normalized to [0,1]
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution for efficient computation.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                 stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class ChannelAttention(nn.Module):
    """
    Channel attention using squeeze-and-excite mechanism.
    """
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """
    Spatial attention using lightweight 3x3 convolution.
    """
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class UltraLightAttention(nn.Module):
    """
    Ultra-Light Attention (ULA) module with entropy-based pruning.
    Combines channel and spatial attention mechanisms.
    """
    def __init__(self, in_channels, entropy_threshold=0.1):
        super(UltraLightAttention, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()
        self.entropy_threshold = entropy_threshold
    
    def compute_entropy(self, attention_map):
        """Compute entropy of attention map for pruning."""
        # Normalize attention map to probability distribution
        attention_map = attention_map.view(attention_map.size(0), -1)
        attention_map = F.softmax(attention_map, dim=1)
        
        # Compute entropy
        entropy = -torch.sum(attention_map * torch.log(attention_map + 1e-8), dim=1)
        return entropy.mean()
    
    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        ca_entropy = self.compute_entropy(ca)
        
        # Apply channel attention if entropy is above threshold
        if ca_entropy > self.entropy_threshold:
            x = x * ca
        
        # Spatial attention
        sa = self.spatial_attention(x)
        sa_entropy = self.compute_entropy(sa)
        
        # Apply spatial attention if entropy is above threshold
        if sa_entropy > self.entropy_threshold:
            x = x * sa
        
        return x

class HALOUNetBlock(nn.Module):
    """
    HALO-UNet block using depthwise separable convolutions.
    """
    def __init__(self, in_channels, out_channels):
        super(HALOUNetBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class HALO_UNet(nn.Module):
    """
    Hardware Aware Lightweight Optimized U-Net (HALO-UNet) with Ultra-Light Attention.
    Uses depthwise separable convolutions for efficiency.
    """
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        super(HALO_UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down part of HALO-UNet (Encoder)
        for feature in features:
            self.downs.append(HALOUNetBlock(in_channels, feature))
            in_channels = feature
        
        # Bottleneck with ULA
        self.bottleneck = HALOUNetBlock(features[-1], features[-1]*2)
        self.ula = UltraLightAttention(features[-1]*2)
        
        # Up part of HALO-UNet (Decoder)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(HALOUNetBlock(feature*2, feature))
        
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck with ULA
        x = self.bottleneck(x)
        x = self.ula(x)  # Apply Ultra-Light Attention
        
        skip_connections = skip_connections[::-1]
        
        # Decoder path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Transposed convolution
            skip_connection = skip_connections[idx//2]
            
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)  # HALO-UNet block
        
        return self.final_conv(x)

# Legacy UNET for compatibility
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
            
        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2),
            )
            self.ups.append(DoubleConv(feature*2, feature))
            
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        return self.final_conv(x)

def test():
    # Test HALO-UNet
    x = torch.randn(2, 3, 256, 256)
    model = HALO_UNet(in_channels=3, out_channels=1)          
    preds = model(x)
    print(f"HALO-UNet - Input shape: {x.shape}, Output shape: {preds.shape}")
    
    # Test Denoising Autoencoder
    autoencoder = DenoisingAutoencoder(in_channels=3)
    denoised = autoencoder(x)
    print(f"Autoencoder - Input shape: {x.shape}, Output shape: {denoised.shape}")
    
    # Test ULA module
    ula = UltraLightAttention(in_channels=64)
    test_features = torch.randn(2, 64, 64, 64)
    attended_features = ula(test_features)
    print(f"ULA - Input shape: {test_features.shape}, Output shape: {attended_features.shape}")

if __name__ == "__main__":
    test()
          
        
        
