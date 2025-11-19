import time
from src.model.model import modelMet
from src.data.data_obtencion import generar_dataloaders, generar_dataset
import torch

device = "cuda:0"

## SE MIDE EL TIEMPO QUE SE TARDA EN EJECUTAR EL FORWARD ##


if __name__ == "__main__":
    modelo = modelMet().to(device)

    train_dataset, val_dataset, test_dataset = generar_dataset(
        path_shards="./data/raw/shards/*",
        props=[0.85, 0.1, 0.05])
    
    train_loader = generar_dataloaders(train_dataset, split="train", batch_size=128, num_workers=6)

    val_loader = generar_dataloaders(val_dataset, split = "val", batch_size= 128, num_workers= 6)

    modelo.eval()
    start_train = time.time()
    print("Midiendo tiempo medio por batch en train...")

    with torch.no_grad():
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            pred = modelo(x)
            if i == 9:   
                break
        

    end_train = time.time()
    tiempo_batch_train = (end_train - start_train) / 10.0
    print(f"Tiempo medio por batch en train ≈ {tiempo_batch_train} s")

    start_val = time.time()
    print("Midiendo tiempo medio por batch en val...")
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            pred = modelo(x)
            if i == 9:   
                break

    end_val = time.time()
    tiempo_batch_val = (end_val - start_val) / 10.0
    print(f"Tiempo medio por batch en val ≈ {tiempo_batch_val} s")


    print("Contando número de batches en train_loader")
    num_batches = 0
    for _ in train_loader:
        num_batches += 1

    print("Contando número de batches en val_loader")
    num_batches_val = 0
    for _ in val_loader:
        num_batches_val += 1


    print(f"Número de batches en train: {num_batches}")
    print(f"Número de batches en val: {num_batches_val}")

    tiempo_epoca = tiempo_batch_train * num_batches + tiempo_batch_val * num_batches_val
    print(f"Tiempo estimado por época ≈ {tiempo_epoca} s "
          f"({tiempo_epoca/60} minutos)")