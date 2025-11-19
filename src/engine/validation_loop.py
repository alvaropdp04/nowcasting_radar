import torch
import torch.nn as nn



device = "cuda:0"


def val_loop(model, loss_module, val_dataloader):

    val_loss = 0.0
    num_batches_val = 0
    model.eval()
    with torch.no_grad():
        for ventana, frame in val_dataloader:
            ventana, frame = ventana.to(device, dtype = torch.float32), frame.to(device, dtype = torch.float32)
            pred = model(ventana)
            loss =  loss_module(pred, frame, a = 5, b = 15)
            val_loss += loss.item()
            num_batches_val +=1
            
    return val_loss / num_batches_val


