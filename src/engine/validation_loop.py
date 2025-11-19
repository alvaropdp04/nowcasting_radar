import torch
import torch.nn as nn
from torch.amp import autocast



device = "cuda:0"

def val_loop(model, loss_module, val_dataloader, use_amp=True):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for ventana, frame in val_dataloader:
            ventana = ventana.to(device, dtype=torch.float32)
            frame   = frame.to(device, dtype=torch.float32)

            if use_amp:
                with autocast(device_type="cuda", dtype=torch.float16):
                    pred = model(ventana)
                    loss = loss_module(pred, frame, a=5, b=15)
            else:
                pred = model(ventana)
                loss = loss_module(pred, frame, a=5, b=15)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


