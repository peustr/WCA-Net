import torch
import torch.nn.functional as f


def grad_find(model, data, target, sample_size=30):
    data.requires_grad = True
    ensemble_xs = data.repeat([sample_size, 1, 1, 1])
    labels = target.repeat([sample_size])
    logits = model(ensemble_xs)

    loss = f.cross_entropy(logits, labels)
    loss.backward()

    gradient = data.grad
    data.requires_grad = False

    return gradient


def pgd(model, data, target, epsilon=8./255., k=10, a=0.01, random_start=True, d_min=0, d_max=1):
    model.eval()
    # perturbed_data = copy.deepcopy(data)
    perturbed_data = data.clone()

    data_max = data + epsilon
    data_min = data - epsilon
    data_max.clamp_(d_min, d_max)
    data_min.clamp_(d_min, d_max)

    if random_start:
        with torch.no_grad():
            perturbed_data.data = data + perturbed_data.uniform_(-1 * epsilon, epsilon)
            perturbed_data.data.clamp_(d_min, d_max)

    for _ in range(k):
        gradient = grad_find(model, perturbed_data, target)

        with torch.no_grad():
            perturbed_data.data += a * torch.sign(gradient)
            perturbed_data.data = torch.max(torch.min(perturbed_data, data_max), data_min)

    return perturbed_data
