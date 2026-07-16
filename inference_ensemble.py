import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from config import Config
from dataset.advancedFD import AdaptiveFloodDataset
from models.unet_fno import HybridUNetFNO

def load_ensemble_models(checkpoint_dir, num_models, device, config):
    models = []
    for i in range(num_models):
        ckpt_path = os.path.join(checkpoint_dir, f"model_seed_{i}.pt")
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint {ckpt_path} not found. Skipping.")
            continue
            
        model = HybridUNetFNO(in_channels=5, width=config.MODEL_WIDTH, modes=config.MODEL_MODES).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Handle DDP state dict keys if necessary
        state_dict = checkpoint.get("model_state", checkpoint)
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(clean_state_dict)
        model.eval()
        models.append(model)
        
    print(f"Successfully loaded {len(models)} models for the ensemble.")
    return models

def main():
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load the dataset (using a small subset for demonstration)
    base_data_path = '/home/tem/flood_mapping/data_Sen1Floods11/v1.1/data/flood_events/WeaklyLabeled'
    if not os.path.exists(base_data_path):
        print(f"Data path {base_data_path} not found. Please update path for actual inference.")
        return

    dataset = AdaptiveFloodDataset(
        sar_dir=os.path.join(base_data_path, 'S1Weak'),
        dem_dir=os.path.join(base_data_path, 'DEM'),
        mask_dir=os.path.join(base_data_path, 'S1OtsuLabelWeak'),
        target_size=(config.IMG_SIZE, config.IMG_SIZE)
    )
    
    # Use the validation split
    total_samples = len(dataset)
    indices = np.arange(total_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    val_indices = indices[int(0.8 * total_samples):]
    val_ds = Subset(dataset, val_indices)
    
    loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # 2. Load Models
    checkpoint_dir = "./checkpoints"
    num_models = 5
    models = load_ensemble_models(checkpoint_dir, num_models, device, config)
    if len(models) < 2:
        print("Not enough models to form an ensemble. Please train multiple models and save them as model_seed_0.pt, etc.")
        return

    # 3. Inference & Uncertainty Disentanglement
    print("Running ensemble inference...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x = batch["image"].to(device)
            
            ensemble_means = []
            ensemble_vars = []
            
            for model in models:
                preds = model(x)
                ensemble_means.append(preds["h_mean"].cpu().numpy())
                # Variance is exp(logvar)
                ensemble_vars.append(np.exp(preds["h_logvar"].cpu().numpy()))
                
            # Convert to numpy arrays: shape (M, Batch, 1, H, W)
            ensemble_means = np.stack(ensemble_means)
            ensemble_vars = np.stack(ensemble_vars)
            
            # -- Law of Total Variance --
            # 1. Aleatoric Uncertainty: Mean of the predicted variances
            aleatoric = np.mean(ensemble_vars, axis=0)
            
            # 2. Epistemic Uncertainty: Variance of the predicted means
            epistemic = np.var(ensemble_means, axis=0)
            
            # 3. Total Predictive Variance
            total_variance = aleatoric + epistemic
            
            print(f"Sample {batch_idx}:")
            print(f"  Mean Aleatoric: {aleatoric.mean():.6f}")
            print(f"  Mean Epistemic: {epistemic.mean():.6f}")
            print(f"  Epistemic Contribution: {(epistemic.sum() / total_variance.sum()) * 100:.4f}%\n")
            
            if batch_idx >= 2: # Just process a few for the demo
                break

if __name__ == "__main__":
    main()
