import torch
import torch.nn.functional as F

def calculate_metrics(pred_h, pred_u, pred_v, y_h, y_u, y_v, threshold=0.05):
    pred_mask = (pred_h > threshold).float()
    true_mask = (y_h > threshold).float()

    intersection = (pred_mask * true_mask).sum()
    union = (pred_mask + true_mask).sum() - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)

    rmse_h = torch.sqrt(F.mse_loss(pred_h, y_h))
    rmse_u = torch.sqrt(F.mse_loss(pred_u, y_u))
    rmse_v = torch.sqrt(F.mse_loss(pred_v, y_v))

    return iou.item(), rmse_h.item(), rmse_u.item(), rmse_v.item()
