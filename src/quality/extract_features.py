"""
src/quality/extract_features.py
Resumable 512-d global-pooled ResNet18 feature extractor on CPU for FactoryPulse.
Extracts visual embeddings for all 12,000 images in data/organizer/train.
Saves to cache/features_resnet18.npz with checkpointing for safe resumption.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Callable
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision.models as models
import torchvision.transforms as transforms

ROOT_DIR = str(Path(__file__).resolve().parent.parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def ensure_split_file(cache_dir: str = "cache", data_dir: str = "data/organizer/train", random_state: int = 42) -> pd.DataFrame:
    """
    Ensure stratified 60/20/20 train/val/test split of the 12,000 images exists in cache/split.csv.
    """
    os.makedirs(cache_dir, exist_ok=True)
    split_path = os.path.join(cache_dir, "split.csv")
    if os.path.exists(split_path):
        df = pd.read_csv(split_path)
        if len(df) == 12000 and set(df.columns) >= {"path", "label", "category", "split"}:
            return df

    import glob
    from sklearn.model_selection import train_test_split

    classes = ["normal", "crack", "hole", "rust", "scratch"]
    rows = []
    for c in sorted(classes):
        class_path = os.path.join(data_dir, c)
        files = sorted(glob.glob(os.path.join(class_path, "*.png")))
        lbl = "good" if c == "normal" else "defective"
        for f in files:
            rel_p = os.path.relpath(f, ROOT_DIR).replace("\\", "/")
            rows.append({"path": rel_p, "label": lbl, "category": c})

    df = pd.DataFrame(rows)
    train_df, temp_df = train_test_split(df, test_size=0.40, random_state=random_state, stratify=df["category"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=random_state, stratify=temp_df["category"])

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    final_df = pd.concat([train_df, val_df, test_df]).sort_values(by=["category", "path"]).reset_index(drop=True)
    final_df = final_df[["path", "label", "category", "split"]]
    final_df.to_csv(split_path, index=False)
    return final_df


def extract_features(
    batch_size: int = 64,
    cache_dir: str = "cache",
    data_dir: str = "data/organizer/train",
    progress_callback: Optional[Callable[[float], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None
) -> str:
    """
    Extract 512-d ResNet18 features for all 12,000 images.
    Resumable: Checkpoints partial results to cache/features_resnet18_partial.npz.
    Saves final outputs to cache/features_resnet18.npz.
    """
    os.makedirs(cache_dir, exist_ok=True)
    full_cache_path = os.path.join(cache_dir, "features_resnet18.npz")
    partial_cache_path = os.path.join(cache_dir, "features_resnet18_partial.npz")

    # If full cache already exists, return
    if os.path.exists(full_cache_path):
        try:
            data = np.load(full_cache_path, allow_pickle=True)
            if len(data["features"]) == 12000:
                if status_callback:
                    status_callback(f"Full ResNet18 cache already exists ({len(data['features'])} features).")
                return full_cache_path
        except Exception:
            pass

    split_df = ensure_split_file(cache_dir, data_dir)
    total_images = len(split_df)

    # Initialize PyTorch ResNet18 model and preprocessing pipeline
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    extracted_feats = []
    start_idx = 0

    # Resume from partial checkpoint if available
    if os.path.exists(partial_cache_path):
        try:
            partial_data = np.load(partial_cache_path, allow_pickle=True)
            extracted_feats = list(partial_data["features"])
            start_idx = len(extracted_feats)
            msg = f"Resuming feature extraction from image {start_idx:,} / {total_images:,}..."
            print(msg)
            if status_callback:
                status_callback(msg)
        except Exception as ex:
            print(f"Failed to read partial checkpoint, restarting extraction: {ex}")
            extracted_feats = []
            start_idx = 0

    paths_list = split_df["path"].tolist()
    labels_list = split_df["label"].tolist()
    categories_list = split_df["category"].tolist()
    splits_list = split_df["split"].tolist()

    t_start = time.time()
    save_every_batches = 10
    batch_count = 0

    for i in range(start_idx, total_images, batch_size):
        batch_paths = paths_list[i : i + batch_size]
        batch_abs = [os.path.join(ROOT_DIR, p) if not os.path.isabs(p) else p for p in batch_paths]
        
        batch_tensors = torch.stack([preprocess(Image.open(p)) for p in batch_abs])
        with torch.no_grad():
            feats = model(batch_tensors).squeeze().numpy()
            if len(batch_paths) == 1:
                feats = feats.reshape(1, -1)
        
        for f_row in feats:
            extracted_feats.append(f_row)

        curr_done = len(extracted_feats)
        batch_count += 1
        pct = curr_done / total_images
        elapsed = time.time() - t_start
        fps = (curr_done - start_idx) / max(1e-4, elapsed)

        if progress_callback:
            progress_callback(pct)
        if status_callback:
            status_callback(f"Extracting ResNet18 features: {curr_done:,}/{total_images:,} ({fps:.1f} img/s)...")

        # Checkpoint partial progress
        if batch_count % save_every_batches == 0:
            np.savez_compressed(
                partial_cache_path,
                features=np.array(extracted_feats, dtype=np.float32),
                paths=np.array(paths_list[:curr_done]),
                labels=np.array(labels_list[:curr_done]),
                categories=np.array(categories_list[:curr_done]),
                splits=np.array(splits_list[:curr_done])
            )

    # Final complete save
    final_feats = np.array(extracted_feats, dtype=np.float32)
    np.savez_compressed(
        full_cache_path,
        features=final_feats,
        paths=np.array(paths_list),
        labels=np.array(labels_list),
        categories=np.array(categories_list),
        splits=np.array(splits_list)
    )

    # Clean up partial checkpoint
    if os.path.exists(partial_cache_path):
        try:
            os.remove(partial_cache_path)
        except Exception:
            pass

    t_total = time.time() - t_start
    print(f"Extraction complete! Saved {len(final_feats):,} features to {full_cache_path} in {t_total:.1f}s.")
    return full_cache_path


if __name__ == "__main__":
    extract_features()
