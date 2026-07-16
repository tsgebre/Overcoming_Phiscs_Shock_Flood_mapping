import os
import torch
import torch.distributed as dist
import numpy as np
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter

from config import Config
from dataset.advancedFD import AdaptiveFloodDataset
from trainer import Trainer

def main():
    # 1. Load config and local rank
    conf = Config()
    conf.LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(conf.LOCAL_RANK)

    # Initialize distributed process group
    dist.init_process_group(backend="nccl", init_method="env://")

    print(f"Project: {conf.PROJECT_NAME} | Mode: {conf.MODE}")

    # 2. Setup dataset
    base_data_path = '/home/tem/flood_mapping/data_Sen1Floods11/v1.1/data/flood_events/WeaklyLabeled'
    if not os.path.exists(base_data_path):
        print(f"Warning: Data path {base_data_path} not found.")
        # Returning here for safety if path doesn't exist, but you might want to mock data or exit properly.
        # return

    try:
        dataset = AdaptiveFloodDataset(
            sar_dir=os.path.join(base_data_path, 'S1Weak'),
            dem_dir=os.path.join(base_data_path, 'DEM'),
            mask_dir=os.path.join(base_data_path, 'S1OtsuLabelWeak'),
            target_size=(conf.IMG_SIZE, conf.IMG_SIZE),
            manning_n=getattr(conf, "MANNING_N", 0.03)
        )

        # Deterministic 80/20 split
        total_samples = len(dataset)
        indices = np.arange(total_samples)
        np.random.seed(42)
        np.random.shuffle(indices)
        split_idx = int(0.8 * total_samples)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_ds = Subset(dataset, train_indices)
        val_ds = Subset(dataset, val_indices)

        print(f"Training Samples: {len(train_ds)} | Validation Samples: {len(val_ds)}")
        os.makedirs(conf.OUT_DIR, exist_ok=True)

        trainer = Trainer(conf, train_ds, val_ds)
        trainer.train()
    except Exception as e:
        print(f"Initialization error: {e}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
