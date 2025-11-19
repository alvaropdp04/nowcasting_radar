import torch
import numpy as np
import torch.nn as nn
from piqa import MS_SSIM

device = "cuda:0"

bins_base = torch.tensor([-10., 0., 10., 20., 30., 40., 50., 80.], dtype=torch.float32, device=device)
pesos_base = torch.tensor([0.01, 0.08, 0.80, 1.00, 3.00, 12.0, 70.0], dtype=torch.float32, device=device)

ms_ssim_module = MS_SSIM(value_range = 1.0, n_channels=1) 
ms_ssim_module = ms_ssim_module.to(device)
loss_mse = nn.MSELoss(reduction="none")

def loss_radar(pred, target, a = 5.0, b = 15.0):
    bins  = bins_base.to(target.device)
    pesos = pesos_base.to(target.device)

    pred_norm   = pred.clamp(0.0, 1.0)
    target_norm = target.clamp(0.0, 1.0)

    bins_normalizados = (bins + 10.0) / 90.0

    indices = torch.bucketize(target_norm, boundaries=bins_normalizados[1:])
    indices = torch.clamp(indices, max=len(pesos) - 1)
    matriz_pesos = pesos[indices]

    score_mse    = loss_mse(pred_norm, target_norm)
    score_msssim = 1 - ms_ssim_module(pred_norm, target_norm)

    loss_final = a * (matriz_pesos * score_mse).mean() + b * score_msssim
    return loss_final
    


