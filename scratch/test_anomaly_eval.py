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
from sklearn.metrics import roc_auc_score, roc_curve

def run_anomaly_eval():
    split_df = pd.read_csv("cache/split.csv")
    
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
    
    # 1. Build or load memory bank
    bank_cache = "cache/patch_memory_bank.pt"
    if os.path.exists(bank_cache):
        bank_data = torch.load(bank_cache)
        bank = bank_data["bank"]
        proj_matrix = bank_data["proj_matrix"]
        print(f"Loaded memory bank from {bank_cache}, shape: {bank.shape}")
    else:
        print("Building memory bank from 300 normal train images...")
        normal_train = split_df[(split_df["split"] == "train") & (split_df["category"] == "normal")].head(300)
        patch_list = []
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
        all_patches = torch.cat(patch_list, dim=0)
        torch.manual_seed(42)
        perm = torch.randperm(all_patches.size(0))[:30000]
        bank = all_patches[perm]
        os.makedirs("cache", exist_ok=True)
        torch.save({"bank": bank, "proj_matrix": proj_matrix}, bank_cache)
        print(f"Saved memory bank to {bank_cache}, shape: {bank.shape}")
        
    def score_single(img_tensor):
        with torch.no_grad():
            _ = model(img_tensor.unsqueeze(0))
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
            smoothed = gaussian_filter(map_np, sigma=4.0)
            score = float(np.max(smoothed))
            return score, smoothed

    # 2. Compute anomaly scores on NORMAL VAL images (480 images, or subset)
    normal_val = split_df[(split_df["split"] == "val") & (split_df["category"] == "normal")]["path"].tolist()
    print(f"Scoring {len(normal_val)} normal val images for 99th percentile threshold...")
    val_scores = []
    t_val0 = time.time()
    for p in normal_val:
        img = Image.open(p)
        score, _ = score_single(preprocess(img))
        val_scores.append(score)
    print(f"Scored {len(val_scores)} normal val images in {time.time() - t_val0:.2f} s")
    threshold_99 = float(np.percentile(val_scores, 99.0))
    print(f"99th Percentile Normal Val Threshold: {threshold_99:.4f}")
    
    # 3. Sanity check: fraction of normal images below threshold
    # On val:
    frac_val_normal_below = float(np.mean(np.array(val_scores) <= threshold_99))
    print(f"Sanity Check (VAL Normal): {frac_val_normal_below*100:.2f}% stay below threshold (expected >= 99%)")
    
    # 4. Report image-level ROC-AUC on 500-image test subset (100 per class)
    print("Scoring 500-image test subset (100 per class)...")
    test_subset_records = []
    for c in ["normal", "crack", "hole", "rust", "scratch"]:
        df_c = split_df[(split_df["split"] == "test") & (split_df["category"] == c)].head(100)
        test_subset_records.append(df_c)
    test_500 = pd.concat(test_subset_records, ignore_index=True)
    
    t_test0 = time.time()
    test_scores = []
    test_labels_binary = []
    test_normal_scores = []
    
    for _, row in test_500.iterrows():
        p = row["path"]
        c = row["category"]
        is_def = 0 if c == "normal" else 1
        img = Image.open(p)
        score, _ = score_single(preprocess(img))
        test_scores.append(score)
        test_labels_binary.append(is_def)
        if is_def == 0:
            test_normal_scores.append(score)
            
    print(f"Scored 500 test images in {time.time() - t_test0:.2f} s ({time.time() - t_test0 / 500:.4f} s/img)")
    
    roc_auc = float(roc_auc_score(test_labels_binary, test_scores))
    print(f"500-Image Test Subset Anomaly ROC-AUC: {roc_auc:.4f} ({roc_auc*100:.2f}%)")
    
    # Test Normal sanity check
    frac_test_normal_below = float(np.mean(np.array(test_normal_scores) <= threshold_99))
    print(f"Sanity Check (TEST Normal): {frac_test_normal_below*100:.2f}% stay below threshold")

if __name__ == "__main__":
    run_anomaly_eval()
