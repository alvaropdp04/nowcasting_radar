import time
from src.model.model import modelMet
from src.data.data_obtencion import generar_dataloaders, generar_dataset
import torch

device = "cuda:0"

if __name__ == "__main__":
    modelo = modelMet().to(device)
    train_dataset, val_dataset, test_dataset = generar_dataset(
        path_shards="./data/raw/shards/*",
        props=[0.85, 0.1, 0.05]
    )
    train_loader = generar_dataloaders(
        train_dataset, split="train", batch_size=8, num_workers=8
    )

    modelo.eval()
    start = time.time()
    print("Midiendo tiempo medio por batch...")

    with torch.no_grad():
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            pred = modelo(x)
            if i == 9:   
                break

    end = time.time()
    tiempo_batch = (end - start) / 10.0
    print(f"Tiempo medio por batch ≈ {tiempo_batch} s")

    print("Contando número de batches en train_loader")
    num_batches = 0
    for _ in train_loader:
        num_batches += 1

    print("Número de batches por época:", num_batches)

    tiempo_epoca = tiempo_batch * num_batches
    print(f"Tiempo estimado por época ≈ {tiempo_epoca} s "
          f"({tiempo_epoca/60} minutos)")