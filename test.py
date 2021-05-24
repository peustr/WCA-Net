import torch

from foolbox import PyTorchModel
from foolbox.attacks import FGSMMC, PGDMC, BIMMC, CWMC


attacks = {
    'FGSM': FGSMMC(),
    'PGD': PGDMC(rel_stepsize=0.1, steps=10),
    'BIM': BIMMC(rel_stepsize=0.1, steps=1000),
    'C&W': CWMC(steps=1000, stepsize=5e-4, confidence=0., initial_const=1e-3),
}


def test_attack(model, data_loader, attack_name, epsilon_values, args, device='cpu'):
    model.eval()
    attack_model = attacks[attack_name]
    # For adversarial testing, the pre-processing happens in the foolbox wrapper.
    if args['dataset'] in ('mnist', 'fmnist'):
        preprocessing = None
    elif args['dataset'] == 'cifar10':
        preprocessing = dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010), axis=-3)
    elif args['dataset'] == 'cifar100':
        preprocessing = dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761), axis=-3)
    elif args['dataset'] == 'svhn':
        preprocessing = dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), axis=-3)
    else:
        raise NotImplementedError('Dataset not supported.')
    fbox_model = PyTorchModel(model, bounds=(0, 1), device=device, preprocessing=preprocessing)
    success_cum = []
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        if attack_name in ('FGSM', 'PGD', 'BIM', 'C&W'):
            advs, _, success = attack_model(
                fbox_model, data, target, epsilons=epsilon_values, mc=args['monte_carlo_runs'])
        elif attack_name in ('Few-Pixel',):
            # Since the few-pixel attack is not supported by foolbox, we have a different testing pipeline.
            raise NotImplementedError()
        success_cum.append(success)
    success_cum = torch.cat(success_cum, dim=1)
    robust_accuracy = 1 - success_cum.float().mean(axis=-1)
    return robust_accuracy
