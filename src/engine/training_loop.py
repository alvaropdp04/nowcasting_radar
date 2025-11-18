import torch
import torch.nn as nn
from src.model.model import modelMet
from src.engine.validation_loop import val_loop


device = "cuda:0"

def train_loop(model, optimizer, train_dataloader, val_dataloader, loss_module, num_epoches = 100, patience = 15):
    min_loss = float('inf')
    no_improvement = 0

    for epoch in range(num_epoches):
        model.train()
        loss_epoch_train = 0.0
        for ventana, frame in train_dataloader:
            ventana, frame = ventana.to(device, dtype = torch.float32), frame.to(device, dtype = torch.float32)
            optimizer.zero_grad()
            pred = model(ventana)
            loss = loss_module(pred, frame, a = 5, b = 15)
            loss.backward()
            optimizer.step()
            loss_epoch_train += loss.item()
        val_loss = val_loop(model = model, loss_module = loss_module, val_dataloader= val_dataloader)
        loss_epoch_train = loss_epoch_train/len(train_dataloader)
        print(f"Época {epoch+1} finalizada. RESULTADOS: Loss en train: {loss_epoch_train}. Loss en validación: {val_loss}")

        if val_loss < min_loss:
            print("Durante esta época se ha conseguido reducir el error en validación")
            min_loss = val_loss
            no_improvement = 0
            best_state_dict = model.state_dict()
        else:
            no_improvement += 1
            print(f"No se ha mejorado durante esta época. Si se alcanza dicho contador se finalizará el entrenamiento: {no_improvement}/{patience}")
        
        if no_improvement >= patience:
            print(f"ENTRENAMIENTO FINALIZADO POR PARADA TEMPRANA EN LA ÉPOCA {epoch+1}")
            break
        
    return min_loss, best_state_dict





