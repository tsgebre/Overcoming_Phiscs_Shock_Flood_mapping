# Overcoming 'Physics Shock' in Earth Observation

Official PyTorch implementation of the paper: **"Overcoming 'Physics Shock' in Earth Observation: A Heteroscedastic Uncertainty Framework for PINN-based Flood Inference."**

This repository contains the code for the **Uncertainty-Aware PINN** (Attention-Gated FNO-UNet), which dynamically relaxes physical constraints in regions of high sensor noise (aleatoric uncertainty) to stabilize training on noisy SAR data.

## 1. Environment Setup

```bash
git clone https://github.com/your-username/Flood_Physics_Guided_DL.git
cd Flood_Physics_Guided_DL
pip install -r requirements.txt
```

## 2. Dataset Preparation
We use the [Sen1Floods11](http://dx.doi.org/10.1109/CVPRW50498.2020.00113) dataset. Arrange the downloaded data as follows:
```text
/data/Sen1Floods11/v1.1/data/flood_events/WeaklyLabeled/
  └── S1Weak/             # SAR VH/VV GeoTIFFs
  └── DEM/                # Copernicus 30m DEM GeoTIFFs
  └── S1OtsuLabelWeak/    # Weakly labeled flood masks
```
Update `base_data_path` in `main.py` to point to your data directory.

## 3. Training the Model
The training script uses Distributed Data Parallel (DDP). To train on a single machine with 2 GPUs:

```bash
torchrun --nproc_per_node=2 main.py
```

*Note: The Dynamic Warm-Start protocol disables the physics loss for the first 5 epochs (configurable via `WARMUP_EPOCHS` in `config.py`) to prevent gradient explosion ('Physics Shock').*

## 4. Inference & Deep Ensembles
To reproduce the uncertainty disentanglement results (Aleatoric vs. Epistemic), train multiple models with different seeds, then run the ensemble script:

```bash
# Example command (requires inference_ensemble.py)
python inference_ensemble.py --checkpoint_dir ./checkpoints/ --output_dir ./results/
```

## Citation
If you find this code useful, please cite our paper:
```bibtex
@article{gebre2024physics,
  title={Overcoming 'Physics Shock' in Earth Observation: A Heteroscedastic Uncertainty Framework for PINN-based Flood Inference},
  author={Gebre, Tewodros Syum and Talreja, Jagrati and Hashemi-Beni, Leila},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2026}
}
```
```
