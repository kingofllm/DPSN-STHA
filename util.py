import torch
import numpy as np

def masked_mae(pred, real, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(real)
    else:
        mask = (real != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(pred - real)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(pred, real, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(real)
    else:
        mask = (real != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((pred - real) / real)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss) * 100

def masked_mse(pred, real, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(real)
    else:
        mask = (real != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (pred-real)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(pred, real, null_val=np.nan):
    return torch.sqrt(masked_mse(pred, real, null_val))


def computer_loss(prediction, target):
    mae = masked_mae(prediction, target, 0.0).item()
    mape = masked_mape(prediction, target, 0.0).item()
    rmse = masked_rmse(prediction, target, 0.0).item()
    return mae, mape, rmse





