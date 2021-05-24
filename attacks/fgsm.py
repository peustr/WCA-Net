import torch
import torch.nn.functional as f


def fgsm(model, data, target, epsilon=0.03, data_min=0, data_max=1):
    model.eval()
    perturbed_data = data.clone()
    perturbed_data.requires_grad = True
    output = model(perturbed_data)
    loss = f.cross_entropy(output, target)
    if perturbed_data.grad is not None:
        perturbed_data.grad.data.zero_()
    loss.backward()
    sign_data_grad = perturbed_data.grad.data.sign()
    perturbed_data.requires_grad = False
    with torch.no_grad():
        perturbed_data += epsilon * sign_data_grad
        perturbed_data.clamp_(data_min, data_max)
    return perturbed_data
