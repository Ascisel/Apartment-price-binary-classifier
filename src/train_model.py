import numpy as np
import torch
from typing import Any
from src.metrics import get_accuracy
from src.eval_model import get_preds

def train_model(
    model: Any,
    train_loader: torch.utils.data.dataloader.DataLoader,
    loss_fn: Any,
    epoch: int,
    optimizer: Any,
    device: str
) -> Any:

    model.train()
    epoch_losses = []
    for num_x, cat_x, labels in iter(train_loader):
        num_x, cat_x, labels = num_x.to(device), cat_x.to(device), labels.to(device)
        model.train()
        out = model(num_x, cat_x).squeeze()
        loss = loss_fn(out, labels)
        loss.backward()
        epoch_losses.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()
    loss_mean = np.array(epoch_losses).mean()
    print(f'Epoch {epoch} loss {loss_mean:.3}')
    return model

def validate(
    model: Any,
    val_loader: torch.utils.data.dataloader.DataLoader,
    device: str
)-> float:

    preds, labels = get_preds(model, val_loader, device)
    test_acc = get_accuracy(preds, labels)
    return test_acc