import torch
from src.engine.validation_loop import val_loop
from torch.amp import autocast, GradScaler



device = "cuda:0"

def train_loop(model, optimizer, train_dataloader, val_dataloader, loss_module, num_epoches = 100, patience = 15):
    model.to(device)
    
    scaler = GradScaler()  # <- para AMP

    min_loss = float('inf')
    no_improvement = 0
    best_state_dict = None

    for epoch in range(num_epoches):
        model.train()
        loss_epoch_train = 0.0
        num_batches_train = 0

        for ventana, frame in train_dataloader:
            ventana = ventana.to(device, dtype=torch.float32)
            frame   = frame.to(device, dtype=torch.float32)

            optimizer.zero_grad()

            with autocast(device_type="cuda", dtype=torch.float16):
                pred = model(ventana)
                loss = loss_module(pred, frame, a=5, b=15)

            loss_epoch_train += loss.item()
            num_batches_train += 1

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        loss_epoch_train = loss_epoch_train / num_batches_train

        val_loss = val_loop(
            model=model,
            loss_module=loss_module,
            val_dataloader=val_dataloader,         
            use_amp=True         
        )

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




