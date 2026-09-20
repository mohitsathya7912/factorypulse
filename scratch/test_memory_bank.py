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
from scipy.ndimage import gaussian_filter, label, find_objects

def test_memory_bank_builder():
    split_df = pd.read_csv("cache/split.csv")
    normal_train = split_df[(split_df["split"] == "train") & (split_df["category"] == "normal")].head(350)
    print(f"Found {len(normal_train)} normal train images")
    
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    layer2_out, layer3_out = None, None
    def hook_layer2(m, inp, out): nonlocal layer2_out; layer2_out = out
    def hook_layer3(m, inp, out): nonlocal layer3_out; layer3_out = out
    
    model.layer2.register_forward_hook(hook_layer2)
    model.layer3.register_forward_hook(hook_layer3)
    
    torch.manual_seed(42)
    proj_matrix = torch.randn(384, 128) / np.sqrt(128)
    
    avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    
    patch_list = []
    t0 = time.time()
    # Process in batches of 32
    batch_tensors = []
    for i, path in enumerate(normal_train["path"].tolist()[:300]):
        img = Image.open(path)
        batch_tensors.append(preprocess(img))
        if len(batch_tensors) == 32 or i == 299:
            batch = torch.stack(batch_tensors)
            batch_tensors = []
            with torch.no_grad():
                _ = model(batch)
                l3_up = F.interpolate(layer3_out, size=layer2_out.shape[-2:], mode="bilinear", align_corners=False)
                feats = torch.cat([layer2_out, l3_up], dim=1)
                pooled = avgpool(feats)
                B, C, H, W = pooled.shape
                patches = pooled.permute(0, 2, 3, 1).reshape(-1, C)
                proj = torch.matmul(patches, proj_matrix)
                patch_list.append(proj)
    
    all_patches = torch.cat(patch_list, dim=0) # [300 * 1024, 128] = [307200, 128]
    print(f"Extracted {all_patches.shape} patches in {time.time() - t0:.2f} s")
    
    # Subsample to 30,000 random patches
    torch.manual_seed(42)
    perm = torch.randperm(all_patches.size(0))[:30000]
    bank = all_patches[perm]
    print(f"Memory bank shape: {bank.shape}")
    
    # Test on 1 normal test image and 1 crack test image
    normal_test = split_df[(split_df["split"] == "test") & (split_df["category"] == "normal")].iloc[0]["path"]
    crack_test = split_df[(split_df["split"] == "test") & (split_df["category"] == "crack")].iloc[0]["path"]
    
    def score_image(img_path):
        img = Image.open(img_path)
        tensor = preprocess(img).unsqueeze(0)
        t_start = time.time()
        with torch.no_grad():
            _ = model(tensor)
            l3_up = F.interpolate(layer3_out, size=layer2_out.shape[-2:], mode="bilinear", align_corners=False)
            feats = torch.cat([layer2_out, l3_up], dim=1)
            pooled = avgpool(feats)
            B, C, H, W = pooled.shape
            patches = pooled.permute(0, 2, 3, 1).reshape(-1, C)
            proj = torch.matmul(patches, proj_matrix)
            
            dists = torch.cdist(proj, bank)
            min_dists, _ = torch.min(dists, dim=1)
            
            dist_map = min_dists.reshape(1, 1, H, W)
            upsampled = F.interpolate(dist_map, size=(256, 256), mode="bilinear", align_corners=False)
            map_np = upsampled.squeeze().numpy()
            
            # Gaussian smooth
            smoothed = gaussian_filter(map_np, sigma=4.0)
            score = float(np.max(smoothed))
            return score, smoothed, time.time() - t_start

    score_norm, map_norm, t_norm = score_image(normal_test)
    score_crack, map_crack, t_crack = score_image(crack_test)
    
    print(f"Normal Test Image Anomaly Score: {score_norm:.4f} (took {t_norm:.4f}s)")
    print(f"Crack Test Image Anomaly Score:  {score_crack:.4f} (took {t_crack:.4f}s)")

if __name__ == "__main__":
    test_memory_bank_builder()
