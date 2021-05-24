eps_names_mnist = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5']
eps_values_mnist = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
eps_names_cifar = ['  0/255', '  1/255', '  2/255', '  4/255', '  8/255', ' 16/255', ' 32/255', ' 64/255', '128/255']
eps_values_cifar = [0. / 255, 1. / 255, 2. / 255, 4. / 255, 8. / 255, 16. / 255, 32. / 255, 64. / 255, 128. / 255]

dataset_to_attack_strength = {
    'mnist': {'eps_names': eps_names_mnist, 'eps_values': eps_values_mnist},
    'fmnist': {'eps_names': eps_names_mnist, 'eps_values': eps_values_mnist},
    'cifar10': {'eps_names': eps_names_cifar, 'eps_values': eps_values_cifar},
    'cifar100': {'eps_names': eps_names_cifar, 'eps_values': eps_values_cifar},
    'svhn': {'eps_names': eps_names_cifar, 'eps_values': eps_values_cifar},
}

attack_to_dataset_config = {
    'FGSM': dataset_to_attack_strength,
    'PGD': dataset_to_attack_strength,
    'BIM': dataset_to_attack_strength,
    'C&W': {'cifar10': {'eps_names': ['None'], 'eps_values': [None]}},
    'Few-Pixel': {'cifar10': {'eps_names': ['1p', '2p', '3p'], 'eps_values': [1, 2, 3]}},
}


mean_cifar10 = (0.4914, 0.4822, 0.4465)
std_cifar10 = (0.2023, 0.1994, 0.2010)

mean_cifar100 = (0.5071, 0.4867, 0.4408)
std_cifar100 = (0.2675, 0.2565, 0.2761)

mean_generic = (0.5, 0.5, 0.5)
std_generic = (0.5, 0.5, 0.5)


def normalize_cifar10(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean_cifar10[0]) / std_cifar10[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean_cifar10[1]) / std_cifar10[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean_cifar10[2]) / std_cifar10[2]
    return t


def normalize_cifar100(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean_cifar100[0]) / std_cifar100[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean_cifar100[1]) / std_cifar100[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean_cifar100[2]) / std_cifar100[2]
    return t


def normalize_generic(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean_generic[0]) / std_generic[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean_generic[1]) / std_generic[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean_generic[2]) / std_generic[2]
    return t
