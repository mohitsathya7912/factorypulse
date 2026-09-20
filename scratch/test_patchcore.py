import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def test_resnet_layers():
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()
    
    # Check layer output shapes
    x = torch.randn(1, 3, 256, 256)
    
    layer2_out = None
    layer3_out = None
    
    def hook_layer2(module, inp, out):
        nonlocal layer2_out
        layer2_out = out
        
    def hook_layer3(module, inp, out):
        nonlocal layer3_out
        layer3_out = out
        
    h2 = model.layer2.register_forward_hook(hook_layer2)
    h3 = model.layer3.register_forward_hook(hook_layer3)
    
    with torch.no_grad():
        _ = model(x)
        
    print(f"Layer 2 shape: {layer2_out.shape}")  # expected [1, 128, 32, 32]
    print(f"Layer 3 shape: {layer3_out.shape}")  # expected [1, 256, 16, 16]
    
    # 1. Upsample layer3 to layer2 grid (32x32)
    l3_up = F.interpolate(layer3_out, size=layer2_out.shape[-2:], mode="bilinear", align_corners=False)
    print(f"Layer 3 upsampled: {l3_up.shape}")
    
    # 2. Concatenate layer2 and upsampled layer3
    features = torch.cat([layer2_out, l3_up], dim=1) # [1, 384, 32, 32]
    print(f"Concatenated features shape: {features.shape}")
    
    # 3. 3x3 average pooling
    avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    pooled = avgpool(features)
    print(f"3x3 Avg pooled shape: {pooled.shape}")
    
    # 4. Reshape to patches: (B*32*32, 384)
    B, C, H, W = pooled.shape
    patches = pooled.permute(0, 2, 3, 1).reshape(-1, C) # [1024, 384]
    print(f"Patches shape: {patches.shape}")
    
    # 5. Fixed random projection to 128 dims
    torch.manual_seed(42)
    proj_matrix = torch.randn(384, 128) / np.sqrt(128)
    proj_patches = torch.matmul(patches, proj_matrix) # [1024, 128]
    print(f"Projected patches shape: {proj_patches.shape}")
    
    # 6. Test distance calculation against a 30,000 memory bank
    bank = torch.randn(30000, 128)
    
    t0 = time.time()
    # Fast nearest-neighbor using cdist or batched L2 distance
    # torch.cdist(proj_patches, bank) produces (1024, 30000)
    # Memory: 1024 * 30000 * 4 bytes = 122.88 MB RAM - completely fine!
    dists = torch.cdist(proj_patches, bank) # [1024, 30000]
    min_dists, _ = torch.min(dists, dim=1)  # [1024]
    elapsed = time.time() - t0
    print(f"Scoring time for 1024 patches against 30,000 bank: {elapsed:.4f} s")
    
    # Reshape to (32, 32)
    dist_map = min_dists.reshape(1, 1, H, W)
    # Upsample to 256x256
    upsampled_map = F.interpolate(dist_map, size=(256, 256), mode="bilinear", align_corners=False)
    print(f"Upsampled anomaly map shape: {upsampled_map.shape}")

if __name__ == "__main__":
    test_resnet_layers()
