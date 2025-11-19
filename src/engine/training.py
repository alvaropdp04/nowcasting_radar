from src.model.model import modelMet
from src.engine.training_loop import train_loop
from src.data.data_obtencion import generar_dataloaders, generar_dataset
from src.engine.loss import loss_radar
import torch


def run_training():


    device = "cuda:0"

    modelo = modelMet().to(device)
    optimizer = torch.optim.AdamW(modelo.parameters(), lr = 0.001, betas = (0.9, 0.999), weight_decay= 0.01)
    train_dataset,val_dataset,test_dataset = generar_dataset(path_shards= "/content/shards/*", props = [0.85,0.1,0.05])
    train_loader = generar_dataloaders(train_dataset, split = "train", batch_size= 180, num_workers= 8)
    val_loader = generar_dataloaders(val_dataset, split = "val", batch_size= 180, num_workers= 8)
    




    print("Comienza el entrenamiento")
    min_loss, best_model = train_loop(model= modelo, optimizer= optimizer, train_dataloader= train_loader, val_dataloader= val_loader, loss_module = loss_radar, num_epoches= 100, patience = 15)
    return min_loss, best_model