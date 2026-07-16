import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsLoss(nn.Module):
    def __init__(self, config):
        super(PhysicsLoss, self).__init__()
        self.config = config
        self.g = config.GRAVITY
        self.reduction = getattr(config, "LOSS_REDUCTION", "mean")
        
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0

    def get_gradients(self, f):
        grad_x = F.conv2d(f, self.sobel_x.to(f.device), padding=1)
        grad_y = F.conv2d(f, self.sobel_y.to(f.device), padding=1)
        return grad_x, grad_y

    def forward(self, pred_h, pred_u, pred_v, dem, mask):
        hu = pred_h * pred_u
        hv = pred_h * pred_v
        
        dhu_dx, _ = self.get_gradients(hu)
        _, dhv_dy = self.get_gradients(hv)
        continuity_residual = dhu_dx + dhv_dy
        continuity_loss_map = (continuity_residual * mask) ** 2

        wse = pred_h + dem
        dwse_dx, dwse_dy = self.get_gradients(wse)
        smoothness_loss_map = (dwse_dx * mask)**2 + (dwse_dy * mask)**2

        num_water_pixels = mask.sum() + 1.0
        if self.reduction == "mean":
            loss_val = (continuity_loss_map.sum() + smoothness_loss_map.sum()) / num_water_pixels
        else:
            loss_val = continuity_loss_map.sum() + smoothness_loss_map.sum()
            
        return loss_val
