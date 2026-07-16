import os
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.enums import Resampling
from scipy.ndimage import label, generate_binary_structure, median_filter
import random

def get_base_name(filename):
    name = filename.rsplit('.', 1)[0]
    name = name.replace('_aligned_dem', '')
    parts = name.split('_')
    if len(parts) >= 2:
        return '_'.join(parts[:2])
    return name

class AdaptiveFloodDataset(Dataset):
    """
    Adaptive Flood Dataset with patch-based physics-proxy hydrodynamics.
    Produces:
        - 'image': Input features [SAR_VH, SAR_VV, Ratio, DEM_Norm, Slope_Norm]
        - 'mask': Flood binary mask
        - 'h': Depth proxy (meters)
        - 'u': Velocity u-component proxy
        - 'v': Velocity v-component proxy
    """
    def __init__(self, sar_dir, dem_dir, mask_dir,
                 target_size=(256, 256),
                 manning_n=0.03,
                 augment=False):
        self.target_size = target_size
        self.manning_n = manning_n
        self.augment = augment

        # Locate files
        sar_files = sorted([f for f in os.listdir(sar_dir) if f.lower().endswith(".tif")])
        dem_files = sorted([f for f in os.listdir(dem_dir) if f.lower().endswith(".tif")])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(".tif")])

        # Map base names to full paths
        sar_dict = {get_base_name(f): os.path.join(sar_dir, f) for f in sar_files}
        dem_dict = {get_base_name(f): os.path.join(dem_dir, f) for f in dem_files}
        mask_dict = {get_base_name(f): os.path.join(mask_dir, f) for f in mask_files}

        # Intersection of keys
        keys = sorted(list(sar_dict.keys() & dem_dict.keys() & mask_dict.keys()))
        self.files = [(sar_dict[k], dem_dict[k], mask_dict[k]) for k in keys]

        print(f"Dataset loaded with {len(self.files)} samples. Augmentation: {self.augment}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        s_path, d_path, m_path = self.files[idx]

        # 1. Load Data
        with rasterio.open(s_path) as src:
            sar = src.read(out_shape=(src.count, *self.target_size), resampling=Resampling.bilinear)
            if sar.shape[0] > 2: sar = sar[:2, ...]

        with rasterio.open(d_path) as src:
            dem = src.read(1, out_shape=self.target_size, resampling=Resampling.bilinear)

        with rasterio.open(m_path) as src:
            mask = src.read(1, out_shape=self.target_size, resampling=Resampling.nearest)

        # Handle NaNs
        sar = np.nan_to_num(sar)
        dem = np.nan_to_num(dem)
        mask = (mask > 0).astype(np.float32)

        # 1.5 Augmentation
        if self.augment:
            if random.random() > 0.5:
                sar = np.flip(sar, axis=2).copy()
                dem = np.fliplr(dem).copy()
                mask = np.fliplr(mask).copy()
            if random.random() > 0.5:
                sar = np.flip(sar, axis=1).copy()
                dem = np.flipud(dem).copy()
                mask = np.flipud(mask).copy()

        # 2. Input Feature Engineering & Normalization
        if np.mean(sar) > 10.0:
            sar = 10.0 * np.log10(np.clip(sar, 1e-6, None))

        sar[0] = median_filter(sar[0], size=3)
        sar[1] = median_filter(sar[1], size=3)
        ratio = sar[0] - sar[1]

        sar_norm = (sar - np.mean(sar, axis=(1,2), keepdims=True)) / (np.std(sar, axis=(1,2), keepdims=True) + 1e-6)
        dem_norm = (dem - np.mean(dem)) / (np.std(dem) + 1e-6)
        ratio_norm = (ratio - np.mean(ratio)) / (np.std(ratio) + 1e-6)

        dy, dx = np.gradient(dem)
        slope_mag = np.sqrt(dx**2 + dy**2)
        slope_norm = (slope_mag - np.mean(slope_mag)) / (np.std(slope_mag) + 1e-6)

        inputs = np.concatenate([
            sar_norm,
            ratio_norm[None, ...],
            dem_norm[None, ...],
            slope_norm[None, ...]
        ], axis=0).astype(np.float32)

        # 3. Physics Proxy Generation (Targets)
        h = np.zeros_like(dem, dtype=np.float32)
        u = np.zeros_like(dem, dtype=np.float32)
        v = np.zeros_like(dem, dtype=np.float32)

        struct = generate_binary_structure(2, 2)
        labeled, n_patches = label(mask, structure=struct)

        for i in range(1, n_patches + 1):
            region = (labeled == i)
            if region.sum() < 5: continue

            wse_i = np.percentile(dem[region], 95) + 0.1
            h_patch = np.clip(wse_i - dem[region], 0.0, None)
            h[region] = h_patch

            vel_mag = (1.0 / self.manning_n) * (h_patch ** (2/3)) * (slope_mag[region] ** 0.5)
            vel_mag = np.clip(vel_mag, 0, 10.0)

            denom = slope_mag[region] + 1e-6
            nx = -dx[region] / denom
            ny = -dy[region] / denom

            u[region] = vel_mag * nx
            v[region] = vel_mag * ny

        return {
            'image': torch.from_numpy(inputs),
            'mask': torch.from_numpy(mask).long(),
            'h': torch.from_numpy(h).unsqueeze(0),
            'u': torch.from_numpy(u).unsqueeze(0),
            'v': torch.from_numpy(v).unsqueeze(0)
        }
