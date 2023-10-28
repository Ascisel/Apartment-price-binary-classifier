import numpy as np
import torch
from typing import Any
from src.metrics import get_accuracy
from src.eval_model import get_preds

def train_model(
    model: Any,
    train_loader: torch.utils.data.dataloader.DataLoader,
    val_loader: torch.utils.data.dataloader.DataLoader,
    loss_fn: Any,
    epochs: int,
    optimizer: Any,
    device: str
) -> Any:
    
    losses = []
    train_acc = []
    val_acc = []
    for epoch in range(epochs):
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
        losses.append(loss_mean)
        print(f'Epoch {epoch} loss {loss_mean:.3}')
        preds, labels = get_preds(model, train_loader, device)
        train_acc.append(get_accuracy(preds, labels))
        
        # Evaluate
        preds, labels = get_preds(model, val_loader, device)
        test_acc = get_accuracy(preds, labels)
        val_acc.append(test_acc)
        

    print('Final Training Accuracy: {}'.format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))

    return model