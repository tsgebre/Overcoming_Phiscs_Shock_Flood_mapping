import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler
import os
from models.unet_fno import HybridUNetFNO
from losses.physics_swe import PhysicsLoss

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2,3))
        union = probs.sum(dim=(2,3)) + targets.sum(dim=(2,3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

class Trainer:
    def __init__(self, config, train_dataset, val_dataset):
        self.config = config
        self.device = torch.device(f"cuda:{config.LOCAL_RANK}" if torch.cuda.is_available() else "cpu")

        self.train_sampler = DistributedSampler(train_dataset)
        self.val_sampler = DistributedSampler(val_dataset, shuffle=False)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE,
            sampler=self.train_sampler, num_workers=6
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE,
            sampler=self.val_sampler, num_workers=6
        )

        self.best_iou = -1.0
        self.ckpt_path = os.path.join(config.OUT_DIR, "best_model.pt")
        self.last_ckpt_path = os.path.join(config.OUT_DIR, "last_model.pt")

        if hasattr(config, "RESUME") and config.RESUME and os.path.exists(self.last_ckpt_path):
            checkpoint = torch.load(self.last_ckpt_path, map_location=self.device)
            self.model.module.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.best_iou = checkpoint.get("best_iou", -1.0)
            print("🔁 Resumed from last checkpoint")

        width = getattr(config, "MODEL_WIDTH", 64)
        modes = getattr(config, "MODEL_MODES", 16)
        model = HybridUNetFNO(in_channels=5, width=width, modes=modes).to(self.device)
        self.model = DDP(model, device_ids=[config.LOCAL_RANK], output_device=config.LOCAL_RANK) if torch.cuda.is_available() else model

        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.LR)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        self.physics_loss_fn = PhysicsLoss(config).to(self.device)
        self.criterion_mse = nn.MSELoss()
        self.criterion_dice = SoftDiceLoss()
        self.criterion_nll = nn.GaussianNLLLoss()
        self.criterion_bce = nn.BCEWithLogitsLoss()

        self.writer = SummaryWriter(log_dir=os.path.join(config.OUT_DIR, f"logs_rank{config.LOCAL_RANK}"))                       if getattr(config, "USE_TENSORBOARD", False) else None
        self.global_step = 0

    def train(self):
        print(f"Starting training on {self.device}...")
        for epoch in range(self.config.EPOCHS):
            self.model.train()
            total_loss = 0.0
            total_data_loss = 0.0
            total_phys_loss = 0.0

            lambda_phys_current = 0.0 if epoch < self.config.WARMUP_EPOCHS else                 self.config.LAMBDA_PHYS * min(1.0, (epoch - self.config.WARMUP_EPOCHS)/5.0)

            for batch_idx, batch in enumerate(self.train_loader):
                x = batch["image"].to(self.device)
                y_h = batch["h"].to(self.device)
                y_u = batch["u"].to(self.device)
                y_v = batch["v"].to(self.device)
                y_mask = batch["mask"].to(self.device).unsqueeze(1).float()
                z_b = x[:,3:4,:,:]

                self.optimizer.zero_grad()
                preds = self.model(x)

                mask_logit = preds["mask"]
                h_mean = preds["h_mean"]
                h_logvar = preds["h_logvar"]
                pred_u = preds["u"]
                pred_v = preds["v"]

                var = torch.exp(h_logvar)
                loss_h_nll = self.criterion_nll(h_mean, y_h, var)
                loss_u = self.criterion_mse(pred_u, y_u)
                loss_v = self.criterion_mse(pred_v, y_v)
                loss_mask_bce = self.criterion_bce(mask_logit, y_mask)
                loss_mask_dice = self.criterion_dice(mask_logit, y_mask)
                loss_mask = loss_mask_bce + loss_mask_dice

                loss_data = loss_h_nll + loss_u + loss_v + loss_mask

                loss_phys = torch.tensor(0.0, device=self.device)
                if lambda_phys_current > 0.0:
                    loss_phys = self.physics_loss_fn(h_mean, pred_u, pred_v, z_b, y_mask)

                loss = self.config.LAMBDA_DATA * loss_data + lambda_phys_current * loss_phys
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                total_data_loss += loss_data.item()
                total_phys_loss += loss_phys.item()

                if self.writer:
                    self.writer.add_scalar("Train/Total_Loss", loss.item(), self.global_step)
                    self.writer.add_scalar("Train/Data_Loss", loss_data.item(), self.global_step)
                    self.writer.add_scalar("Train/Phys_Loss", loss_phys.item(), self.global_step)
                    self.global_step += 1

            self.scheduler.step()
            avg_loss = total_loss / len(self.train_loader)
            avg_data_loss = total_data_loss / len(self.train_loader)
            avg_phys_loss = total_phys_loss / len(self.train_loader)

            val_metrics = self.validate()
            print(f"Epoch {epoch+1}/{self.config.EPOCHS} | Loss: {avg_loss:.4f} | Val IoU: {val_metrics['iou']:.4f}")

            if val_metrics["iou"] > self.best_iou:
                self.best_iou = val_metrics["iou"]
                if self.config.LOCAL_RANK == 0:
                    model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
                    torch.save({
                        "epoch": epoch,
                        "model_state": model_state,
                        "optimizer_state": self.optimizer.state_dict(),
                        "best_iou": self.best_iou
                    }, self.ckpt_path)
                    print(f"✅ Saved BEST model (IoU: {self.best_iou:.4f})")

        if self.writer:
            self.writer.close()

    def validate(self):
        self.model.eval()
        val_iou = 0.0
        val_rmse_h = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                x = batch["image"].to(self.device)
                y_h = batch["h"].to(self.device)
                y_mask = batch["mask"].to(self.device).unsqueeze(1).float()

                preds = self.model(x)
                mask_logit = preds["mask"]
                h_mean = preds["h_mean"]

                pred_mask = (mask_logit > 0).float()
                intersection = (pred_mask * y_mask).sum()
                union = (pred_mask + y_mask).sum() - intersection
                val_iou += (intersection / union).item() if union > 0 else (1.0 if intersection==0 else 0.0)
                val_rmse_h += torch.sqrt(((h_mean - y_h)**2).mean()).item()

        return {
            "iou": val_iou / len(self.val_loader),
            "rmse_h": val_rmse_h / len(self.val_loader)
        }
