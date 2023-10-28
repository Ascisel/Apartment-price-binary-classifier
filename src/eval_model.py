from typing import Any
import torch
from src.config import EvalConfig as Config



def get_preds(
    model: Any,
    eval_loader: torch.utils.data.dataloader.DataLoader,
    device: str
) -> torch.Tensor:
    
    model.eval()
    for num_x, cat_x, labels in eval_loader:
        num_x, cat_x, labels = num_x.to(device), cat_x.to(device), labels.to(device)
        output = model(num_x, cat_x)

    preds = output > Config.THRESHOLD

    return preds, labels
