import numpy as np


def accuracy(model, data_loader, device='cpu', norm=None):
    positives, total = [], []
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        model.eval()
        if norm is not None:
            data = norm(data)
        logits = model(data)
        positives.append(sum(logits.argmax(-1) == target).item())
        total.append(len(data))
    accuracy = float(np.sum(positives) / np.sum(total))
    return accuracy
