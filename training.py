# Importing required libraries
import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
import wandb
from data_util import CaptionDataset, collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from model import init_model

def train_model(dataset_path, wandb_project, wandb_entity, epochs=50, batch_size=128, learning_rate=5e-4):
    """ Training the model"""

    #setting up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True     # for fastest algorithm from cuDNN

    # Initializing wandb
    wandb.init(project=wandb_project, entity=wandb_entity, config={
        'epochs' : epochs,
        'batch_size' : batch_size,
        'learning_rate' : learning_rate
    })

    # Dataset & Dataloader
    dataset = CaptionDataset(
        root_dir = os.path.join(dataset_path, 'Images'),
        caption_file= os.path.join(dataset_path, 'captions.txt')
    )
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [7000, 1000])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    #  Model Setup
    model = init_model(len(dataset.vocab), device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=2)

    best_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for images, captions, lengths in tqdm(train_loader):
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                outputs = model(images, captions[:, :-1])
                loss = criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    captions[:, 1:].reshape(-1)
                )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

                avg_loss = train_loss / len(train_loader)
                val_loss = validate_model(model, val_loader, criterion, device)

                # wandb logging
                wandb.log({
                    'train_loss' : avg_loss,
                    'val_loss' : val_loss,
                    'lr' : optimizer.param_groups[0]['lr']
                })

                # Checkpoints
                if val_loss < best_loss:
                    best_loss = val_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                # Early stopping
                if early_stop_counter >= 15:
                    print("Early stopping")
                    break

                scheduler.step(val_loss)
    wandb.finish()


def validate_model(model, loader, criterion, device):
    """ Validate the model"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, captions, lengths in loader:
            outputs = model(images.to(device), captions[:, :-1].to(device))
            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                captions[:, 1:].to(device).reshape(-1)
            )
            total_loss += loss.item()
    return total_loss / len(loader)



