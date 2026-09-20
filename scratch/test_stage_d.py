import os
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from scipy.special import softmax

def apply_perturbation(img: Image.Image, p_type: str, seed: int = 42) -> Image.Image:
    if p_type == "orig":
        return img
    elif p_type == "bright_0.6":
        return ImageEnhance.Brightness(img).enhance(0.6)
    elif p_type == "bright_1.4":
        return ImageEnhance.Brightness(img).enhance(1.4)
    elif p_type == "contrast_0.6":
        return ImageEnhance.Contrast(img).enhance(0.6)
    elif p_type == "contrast_1.4":
        return ImageEnhance.Contrast(img).enhance(1.4)
    elif p_type == "rot_90":
        return img.rotate(90, expand=False)
    elif p_type == "rot_180":
        return img.rotate(180, expand=False)
    elif p_type == "hflip":
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif p_type == "blur_1":
        return img.filter(ImageFilter.GaussianBlur(radius=1.0))
    elif p_type == "blur_2":
        return img.filter(ImageFilter.GaussianBlur(radius=2.0))
    elif p_type == "noise_10":
        arr = np.array(img, dtype=np.float32)
        rng = np.random.RandomState(seed)
        noise = rng.normal(0, 10, arr.shape)
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    elif p_type == "noise_25":
        arr = np.array(img, dtype=np.float32)
        rng = np.random.RandomState(seed)
        noise = rng.normal(0, 25, arr.shape)
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    else:
        raise ValueError(f"Unknown perturbation: {p_type}")

def random_augment_train_image(img: Image.Image, seed: int) -> Image.Image:
    rng = np.random.RandomState(seed)
    # Random brightness between 0.7 and 1.3
    b_factor = rng.uniform(0.7, 1.3)
    img = ImageEnhance.Brightness(img).enhance(b_factor)
    # Random contrast between 0.7 and 1.3
    c_factor = rng.uniform(0.7, 1.3)
    img = ImageEnhance.Contrast(img).enhance(c_factor)
    # Random rotation: 0, 90, 180, 270
    rot = rng.choice([0, 90, 180, 270])
    if rot > 0:
        img = img.rotate(rot, expand=False)
    # Random flip
    if rng.rand() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # Random blur
    if rng.rand() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.5)))
    # Random noise
    if rng.rand() > 0.5:
        sigma = rng.uniform(5, 20)
        arr = np.array(img, dtype=np.float32)
        noise = rng.normal(0, sigma, arr.shape)
        img = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    return img

def main():
    print("Testing Stage D Prototype...")
    split_df = pd.read_csv("cache/split.csv")
    
    # 500-image test subset (100 per class)
    test_subset_dfs = []
    classes = ["crack", "hole", "normal", "rust", "scratch"]
    for c in classes:
        df_c = split_df[(split_df["split"] == "test") & (split_df["category"] == c)].head(100)
        test_subset_dfs.append(df_c)
    test_500 = pd.concat(test_subset_dfs, ignore_index=True)
    print(f"500-image test subset: {len(test_500)} images ({test_500['category'].value_counts().to_dict()})")

    # Load ResNet18
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

    # Test extracting 1 perturbation on 500 images
    t0 = time.time()
    tensors = []
    for _, row in test_500.iterrows():
        img = Image.open(row["path"])
        p_img = apply_perturbation(img, "noise_25")
        tensors.append(preprocess(p_img))
    
    batch = torch.stack(tensors)
    with torch.no_grad():
        feats = model(batch).numpy()
    print(f"Extracted 500 noise_25 images in {time.time() - t0:.2f} s, shape: {feats.shape}")

if __name__ == "__main__":
    main()
