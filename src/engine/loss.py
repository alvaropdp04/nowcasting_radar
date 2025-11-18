import torch
import numpy as np
import torch.nn as nn
from piqa import MS_SSIM

def loss_radar(pred, target, a, b):
    bins = torch.tensor([-10.,0.,10.,20.,30.,40.,50.,80.], device = target.device, dtype=target.dtype)
    bins_normalizados = (bins + 10) / 90

    pesos = torch.tensor([0.01,   # [-10,  0) no lluvia / ruido
                          0.08,   # [  0, 10) lluvia muy ligera
                          0.80,   # [ 10, 20) lluvia típica suave   
                          1.00,   # [ 20, 30) lluvia moderada       
                          3.00,   # [ 30, 40) lluvia fuerte
                          12.0,   # [ 40, 50) tormenta fuerte
                          70.0    # [ 50, 80) núcleos / tormentas muy fuertes
                          ], device=target.device, dtype=target.dtype)
    
    loss_mse = nn.MSELoss(reduction = "none")
    ms_ssim = MS_SSIM(data_range = 1.0)
    matriz_pesos = pesos[torch.bucketize(target, boundaries= bins_normalizados) - 1]
    score_mse = loss_mse(pred, target)
    score_msssim = 1 - ms_ssim(pred, target)
    loss_final = a*((matriz_pesos*score_mse).mean()) + b*score_msssim
    return loss_final
    


