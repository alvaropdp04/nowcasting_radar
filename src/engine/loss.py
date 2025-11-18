import torch
import numpy as np
import torch.nn as nn
from piqa import MS_SSIM

device = "cuda:0"

bins_base = torch.tensor([-10., 0., 10., 20., 30., 40., 50., 80.], dtype=torch.float32, device=device)
pesos_base = torch.tensor([0.01, 0.08, 0.80, 1.00, 3.00, 12.0, 70.0], dtype=torch.float32, device=device)

ms_ssim_module = MS_SSIM(data_range=1.0, size_average=True, channel=1) 
ms_ssim_module = ms_ssim_module.to(device)
loss_mse = nn.MSELoss(reduction="none")

def loss_radar(pred, target, a = 5.0, b = 15.0):
    bins = bins_base.to(target.device)
    pesos = pesos_base.to(target.device)

    bins_normalizados = (bins + 10) / 90

    indices = torch.bucketize(target, boundaries=bins_normalizados[1:])
    indices = torch.clamp(indices, max=len(pesos) - 1)
    matriz_pesos = pesos[indices]
    score_mse = loss_mse(pred, target)              
    score_msssim = 1 - ms_ssim_module(pred, target)  
    loss_final = a*(matriz_pesos * score_mse).mean() + b*score_msssim
    return loss_final
    


