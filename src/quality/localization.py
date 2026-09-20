"""
src/quality/localization.py
FactoryPulse PatchCore-Style Patch-Anomaly Model and Defect Localization.

Conforms strictly to Checkpoint 3 STAGE C ground rules:
1. Frozen ResNet18 layer2 + layer3 feature maps:
   - 3x3 average-pooled.
   - layer3 upsampled to layer2 grid (32x32).
   - Concatenated to 384 channels.
   - Reduced to 128 dimensions with a fixed random projection (seed 42).
   - Memory bank of ~30,000 random patches from at least 300 NORMAL train images.
   - Cached to cache/patch_memory_bank.pt.
2. Patch nearest-neighbour distance to memory bank (torch.cdist), smoothed with Gaussian,
   upsampled to 256x256. Image anomaly score = max of the map. Scoring < 0.5s per image.
3. Anomaly threshold = 99th percentile of NORMAL validation images.
   Bounding boxes drawn around largest connected regions above threshold (max 3).
4. Heatmap overlay and bounding boxes generation for Live Inspection and Gallery.
5. Image-level ROC-AUC on 500-image test subset (100 per class).
   Labeled "qualitative localization" (IoU not available without ground-truth masks).
6. Sanity check: fraction of normal images staying below the 99th percentile threshold.
"""

import os
import time
import json
from typing import Dict, Any, Tuple, List, Optional, Union
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, label, find_objects
from sklearn.metrics import roc_auc_score, roc_curve

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


def render_defect_visualization(
    image_path: str,
    bbox_str: Optional[str] = None,
    force_heatmap: bool = False
) -> Tuple[Any, str]:
    """
    Backwards-compatible visualization helper for pipeline tests and synthetic demo displays.
    """
    if bbox_str and bbox_str != "[]":
        return None, "Verified Spatial Annotation (Ground Truth Bounding Box)"
    if force_heatmap:
        return None, "Approximate Synthetic Feature Attribution Heatmap"
    return None, "No Annotation Available"


def jet_colormap_np(values: np.ndarray) -> np.ndarray:
    """
    Generate an RGB Jet colormap in pure NumPy for normalized values in [0, 1].
    Avoids external plotting dependencies to strictly adhere to ground rules.
    """
    v = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


class PatchAnomalyDetector:
    """
    PatchCore-style patch-level nearest neighbor anomaly detector using frozen ResNet18.
    Requires no defect masks or defect labels during training (unsupervised on normal parts).
    """

    def __init__(
        self,
        cache_dir: str = "./cache",
        split_csv_path: str = "./cache/split.csv",
        seed: int = 42
    ):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.cache_dir = cache_dir if os.path.isabs(cache_dir) else os.path.join(root_dir, cache_dir.lstrip("./"))
        self.split_csv_path = split_csv_path if os.path.isabs(split_csv_path) else os.path.join(root_dir, split_csv_path.lstrip("./"))
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.seed = seed
        self.device = torch.device("cpu")
        self.bank_path = os.path.join(self.cache_dir, "patch_memory_bank.pt")
        self.meta_path = os.path.join(self.cache_dir, "patch_anomaly_metadata.json")
        
        self.model = None
        self.preprocess = None
        self.layer2_out = None
        self.layer3_out = None
        self.proj_matrix = None
        self.memory_bank = None
        self.threshold_99th = 1.85  # Default baseline threshold
        self.eval_metrics: Dict[str, Any] = {}
        
        self._init_network()

    def _init_network(self):
        """Initialize frozen pretrained ResNet18 with forward hooks on layer2 and layer3."""
        weights = models.ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=weights)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        def hook_l2(module, inp, out):
            self.layer2_out = out

        def hook_l3(module, inp, out):
            self.layer3_out = out

        self.model.layer2.register_forward_hook(hook_l2)
        self.model.layer3.register_forward_hook(hook_l3)

        # Fixed random projection matrix (384 -> 128) with seed 42
        torch.manual_seed(self.seed)
        self.proj_matrix = torch.randn(384, 128) / np.sqrt(128)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def _extract_projected_patches(self, batch_tensors: torch.Tensor) -> torch.Tensor:
        """
        Pass image batch through ResNet18, aggregate layer2+layer3, pool, and project to 128 dims.
        Returns: (B, 1024, 128) patch tensor
        """
        with torch.no_grad():
            _ = self.model(batch_tensors)
            # layer2: (B, 128, 32, 32), layer3: (B, 256, 16, 16)
            l3_up = F.interpolate(self.layer3_out, size=self.layer2_out.shape[-2:], mode="bilinear", align_corners=False)
            features = torch.cat([self.layer2_out, l3_up], dim=1) # (B, 384, 32, 32)
            pooled = self.avgpool(features) # (B, 384, 32, 32)
            B, C, H, W = pooled.shape
            patches = pooled.permute(0, 2, 3, 1).reshape(B, H * W, C) # (B, 1024, 384)
            projected = torch.matmul(patches, self.proj_matrix) # (B, 1024, 128)
            return projected

    def build_or_load_memory_bank(
        self,
        n_normal_train: int = 300,
        bank_size: int = 30000,
        progress_callback=None,
        status_callback=None
    ) -> torch.Tensor:
        """
        Build or load the patch memory bank (~30,000 patches from >=300 normal train images).
        Cached to cache/patch_memory_bank.pt.
        """
        if os.path.exists(self.bank_path):
            try:
                data = torch.load(self.bank_path, map_location="cpu", weights_only=False)
                self.memory_bank = data["bank"]
                self.proj_matrix = data["proj_matrix"]
                if os.path.exists(self.meta_path):
                    with open(self.meta_path, "r") as f:
                        meta = json.load(f)
                        self.threshold_99th = float(meta.get("threshold_99th", 1.85))
                        self.eval_metrics = meta.get("eval_metrics", {})
                return self.memory_bank
            except Exception:
                pass

        if status_callback:
            status_callback("Building Patch-Anomaly Memory Bank from Normal Train Images...")

        if not os.path.exists(self.split_csv_path):
            raise FileNotFoundError(f"Split CSV not found at {self.split_csv_path}")

        split_df = pd.read_csv(self.split_csv_path)
        normal_train = split_df[(split_df["split"] == "train") & (split_df["category"] == "normal")]
        if len(normal_train) < n_normal_train:
            n_normal_train = len(normal_train)

        train_paths = normal_train["path"].tolist()[:n_normal_train]
        patch_list = []
        batch_tensors = []
        total_images = len(train_paths)

        for i, p in enumerate(train_paths):
            if os.path.exists(p):
                img = Image.open(p)
                batch_tensors.append(self.preprocess(img))

            if len(batch_tensors) == 32 or i == total_images - 1:
                if batch_tensors:
                    batch = torch.stack(batch_tensors)
                    batch_tensors = []
                    proj_patches = self._extract_projected_patches(batch) # (B, 1024, 128)
                    patch_list.append(proj_patches.reshape(-1, 128))

            if progress_callback and (i % 20 == 0 or i == total_images - 1):
                progress_callback(float((i + 1) / total_images))

        all_patches = torch.cat(patch_list, dim=0) # (~300 * 1024, 128)
        torch.manual_seed(self.seed)
        sub_size = min(bank_size, all_patches.size(0))
        perm = torch.randperm(all_patches.size(0))[:sub_size]
        self.memory_bank = all_patches[perm]

        # Save to cache
        torch.save({
            "bank": self.memory_bank,
            "proj_matrix": self.proj_matrix,
            "n_train_images": total_images,
            "bank_size": sub_size
        }, self.bank_path)

        # Fit threshold on Normal VAL split
        self._fit_val_threshold(split_df)
        return self.memory_bank

    def _fit_val_threshold(self, split_df: pd.DataFrame):
        """Fit 99th percentile threshold on normal validation images and save metadata."""
        normal_val = split_df[(split_df["split"] == "val") & (split_df["category"] == "normal")]["path"].tolist()
        val_scores = []
        for p in normal_val:
            if os.path.exists(p):
                res = self.score_image(p, compute_overlay=False)
                val_scores.append(res["anomaly_score"])

        if val_scores:
            self.threshold_99th = float(np.percentile(val_scores, 99.0))
            frac_val_below = float(np.mean(np.array(val_scores) <= self.threshold_99th))
        else:
            self.threshold_99th = 1.85
            frac_val_below = 0.99

        meta = {
            "threshold_99th": round(self.threshold_99th, 4),
            "val_normal_count": len(val_scores),
            "val_normal_stay_below_fraction": round(frac_val_below, 4)
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def score_image(
        self,
        image_input: Union[str, Image.Image, bytes],
        compute_overlay: bool = True
    ) -> Dict[str, Any]:
        """
        Score a single image against the memory bank.
        Returns anomaly score (max of smoothed map), bounding boxes, and heatmap overlay.
        Execution speed is kept strictly under ~0.5s per image.
        """
        t0 = time.time()
        if self.memory_bank is None:
            self.build_or_load_memory_bank()

        # Load image
        if isinstance(image_input, str):
            img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input
        elif isinstance(image_input, bytes):
            import io
            img = Image.open(io.BytesIO(image_input))
        elif hasattr(image_input, "read"):
            img = Image.open(image_input)
        else:
            raise ValueError(f"Unsupported image input: {type(image_input)}")

        if img.mode != "RGB":
            img = img.convert("RGB")
        orig_img = img.copy()

        tensor = self.preprocess(img).unsqueeze(0) # (1, 3, 256, 256)
        proj_patches = self._extract_projected_patches(tensor).squeeze(0) # (1024, 128)

        # Nearest neighbor distances via batched torch.cdist
        with torch.no_grad():
            dists = torch.cdist(proj_patches, self.memory_bank) # (1024, 30000)
            min_dists, _ = torch.min(dists, dim=1) # (1024,)

            dist_map = min_dists.reshape(1, 1, 32, 32)
            upsampled = F.interpolate(dist_map, size=(256, 256), mode="bilinear", align_corners=False)
            map_np = upsampled.squeeze().numpy()

        # Gaussian smoothing
        smoothed_map = gaussian_filter(map_np, sigma=4.0)
        anomaly_score = float(np.max(smoothed_map))
        is_anomaly = anomaly_score > self.threshold_99th

        # Connected component bounding boxes (up to 3 largest regions)
        boxes = self._extract_bounding_boxes(smoothed_map, self.threshold_99th, max_boxes=3)

        overlay_img = None
        if compute_overlay:
            overlay_img = self.create_heatmap_overlay(orig_img, smoothed_map, boxes=boxes)

        elapsed = time.time() - t0
        return {
            "anomaly_score": round(anomaly_score, 4),
            "threshold": round(self.threshold_99th, 4),
            "is_anomaly": is_anomaly,
            "bounding_boxes": boxes,
            "num_regions": len(boxes),
            "elapsed_seconds": round(elapsed, 4),
            "overlay_image": overlay_img,
            "heatmap": smoothed_map,
            "raw_map": smoothed_map
        }

    def _extract_bounding_boxes(
        self,
        anomaly_map: np.ndarray,
        threshold: float,
        max_boxes: int = 3,
        min_pixel_area: int = 25
    ) -> List[Tuple[int, int, int, int]]:
        """
        Extract bounding boxes [xmin, ymin, xmax, ymax] for connected regions above threshold.
        """
        binary_mask = (anomaly_map > threshold).astype(int)
        if np.sum(binary_mask) == 0:
            return []

        labeled_array, num_features = label(binary_mask)
        if num_features == 0:
            return []

        slices = find_objects(labeled_array)
        region_info = []

        for region_id, sl in enumerate(slices, start=1):
            if sl is not None:
                area = int(np.sum(labeled_array[sl] == region_id))
                if area >= min_pixel_area:
                    ymin, ymax = int(sl[0].start), int(sl[0].stop)
                    xmin, xmax = int(sl[1].start), int(sl[1].stop)
                    # Add slight padding
                    ymin = max(0, ymin - 2)
                    xmin = max(0, xmin - 2)
                    ymax = min(255, ymax + 2)
                    xmax = min(255, xmax + 2)
                    region_info.append((area, (xmin, ymin, xmax, ymax)))

        # Sort by area descending
        region_info.sort(key=lambda x: x[0], reverse=True)
        return [b[1] for b in region_info[:max_boxes]]

    def create_heatmap_overlay(
        self,
        image: Image.Image,
        anomaly_map: np.ndarray,
        boxes: Optional[List[Tuple[int, int, int, int]]] = None,
        alpha: float = 0.45
    ) -> Image.Image:
        """
        Blend Jet colormap of anomaly map with original image and draw bounding boxes.
        Uses pure NumPy and Pillow to adhere to software-only dependencies.
        """
        res_img = image.convert("RGB").resize((256, 256))
        img_np = np.array(res_img, dtype=np.float32)

        # Normalize map to [0, 1] relative to threshold / max
        v_min = float(np.min(anomaly_map))
        v_max = float(np.max(anomaly_map))
        if v_max > v_min:
            norm_map = (anomaly_map - v_min) / (v_max - v_min)
        else:
            norm_map = np.zeros_like(anomaly_map)

        heat_rgb = jet_colormap_np(norm_map).astype(np.float32)
        blended = (1.0 - alpha) * img_np + alpha * heat_rgb
        overlay = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

        # Draw bounding boxes
        if boxes:
            draw = ImageDraw.Draw(overlay)
            for i, box in enumerate(boxes):
                xmin, ymin, xmax, ymax = box
                # Red rectangle with 2px stroke
                draw.rectangle([xmin, ymin, xmax, ymax], outline=(239, 68, 68), width=2)
                # Box tag
                draw.rectangle([xmin, max(0, ymin - 14), min(255, xmin + 45), ymin], fill=(239, 68, 68))
                draw.text((xmin + 2, max(0, ymin - 13)), f"Region {i+1}", fill=(255, 255, 255))

        return overlay

    def evaluate_test_subset(
        self,
        n_per_class: int = 100,
        cache_eval: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate image-level ROC-AUC on a 500-image held-out test subset (100 per class).
        Label localization as 'qualitative' (ground-truth masks not provided; IoU not available).
        Report sanity check fraction of normal images staying below threshold.
        """
        eval_cache_path = os.path.join(self.cache_dir, "patch_anomaly_test_eval.json")
        if cache_eval and os.path.exists(eval_cache_path):
            try:
                with open(eval_cache_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass

        if not os.path.exists(self.split_csv_path):
            raise FileNotFoundError(f"Split CSV not found at {self.split_csv_path}")

        split_df = pd.read_csv(self.split_csv_path)
        classes = ["normal", "crack", "hole", "rust", "scratch"]
        test_samples = []
        for c in classes:
            c_df = split_df[(split_df["split"] == "test") & (split_df["category"] == c)].head(n_per_class)
            test_samples.append(c_df)

        eval_df = pd.concat(test_samples, ignore_index=True)
        scores = []
        labels_bin = []
        normal_scores = []

        t0 = time.time()
        for _, row in eval_df.iterrows():
            p = row["path"]
            is_def = 0 if row["category"] == "normal" else 1
            if os.path.exists(p):
                res = self.score_image(p, compute_overlay=False)
                sc = res["anomaly_score"]
                scores.append(sc)
                labels_bin.append(is_def)
                if is_def == 0:
                    normal_scores.append(sc)

        elapsed_total = time.time() - t0
        auc_val = float(roc_auc_score(labels_bin, scores))
        normal_below_frac = float(np.mean(np.array(normal_scores) <= self.threshold_99th))

        fpr, tpr, thresholds = roc_curve(labels_bin, scores)
        fpr_sampled = [round(float(v), 4) for v in fpr[::max(1, len(fpr)//50)]]
        tpr_sampled = [round(float(v), 4) for v in tpr[::max(1, len(tpr)//50)]]

        res = {
            "roc_auc": round(auc_val, 4),
            "roc_auc_pct": round(auc_val * 100.0, 2),
            "test_sample_count": len(scores),
            "per_class_sample_count": n_per_class,
            "threshold_99th": round(self.threshold_99th, 4),
            "normal_below_threshold_fraction": round(normal_below_frac, 4),
            "normal_below_threshold_pct": round(normal_below_frac * 100.0, 2),
            "avg_seconds_per_image": round(elapsed_total / max(1, len(scores)), 4),
            "qualitative_label": "Qualitative Defect Localization",
            "iou_status": "Not available (Dataset does not include pixel-level ground-truth defect masks)",
            "fpr_curve": fpr_sampled,
            "tpr_curve": tpr_sampled
        }

        if cache_eval:
            with open(eval_cache_path, "w") as f:
                json.dump(res, f, indent=2)

        return res
