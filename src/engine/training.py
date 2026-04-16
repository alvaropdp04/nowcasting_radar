import os
import torch
from src.model.model import modelMet
from src.engine.training_loop import train_loop
from src.data.data_obtencion import generar_dataloaders, generar_dataset
from src.engine.loss import loss_radar


def run_training(
    shards_path,
    checkpoint_path,
    h=128,
    batch_size=64,
    lr=0.001,
    num_epoches=100,
    patience=15,
    num_workers=4,
    resume=False,
):
    device = "cuda:0"

    modelo = modelMet(h=h).to(device)

    if resume and os.path.exists(checkpoint_path):
        modelo.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Checkpoint cargado desde {checkpoint_path}")

    optimizer = torch.optim.AdamW(
        modelo.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    train_dataset, val_dataset, _ = generar_dataset(
        path_shards=shards_path, props=[0.85, 0.1, 0.05]
    )
    train_loader = generar_dataloaders(
        train_dataset, split="train", batch_size=batch_size, num_workers=num_workers
    )
    val_loader = generar_dataloaders(
        val_dataset, split="val", batch_size=batch_size, num_workers=num_workers
    )

    print("Comienza el entrenamiento")
    min_loss, best_state_dict = train_loop(
        model=modelo,
        optimizer=optimizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        loss_module=loss_radar,
        num_epoches=num_epoches,
        patience=patience,
        save_path=checkpoint_path,
        scheduler=scheduler,
    )
    return min_loss, best_state_dict
