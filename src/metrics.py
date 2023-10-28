from typing import Any
from torch import Tensor


def get_accuracy(preds: Tensor, labels: Tensor) -> float:

    correct = preds.eq(labels.view_as(preds)).sum().item()
    total = preds.shape[0]
    return correct / total