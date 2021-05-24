import json
import os
import sys

import torch

from attacks.one_pixel import one_pixel_attack
from data_loaders import get_data_loader
from models import model_factory
from test import test_attack
from train import train_vanilla, train_stochastic, train_stochastic_adversarial
from utils import attack_to_dataset_config


def parse_args():
    mode = sys.argv[1]
    if mode not in ('train', 'test', 'train+test'):
        raise ValueError()
    config_file = sys.argv[2]
    with open(config_file, 'r') as fp:
        args = json.loads(fp.read().strip())
    return mode, args


def train(args, device):
    print(args)
    os.makedirs(args['output_path']['stats'], exist_ok=True)
    os.makedirs(args['output_path']['models'], exist_ok=True)
    train_loader = get_data_loader(args['dataset'], args['batch_size'], train=True, shuffle=True, drop_last=True)
    test_loader = get_data_loader(args['dataset'], args['batch_size'], train=False, shuffle=False, drop_last=False)
    model = model_factory(
        args['dataset'], args['training_type'], args['var_type'], args['feature_dim'], args['num_classes'])
    model.to(device)
    if args['pretrained'] is not None:
        if args['pretrained'] not in ('ckpt_best', 'ckpt_last', 'ckpt_robust'):
            raise ValueError('Pre-trained model name must be: [ckpt_best|ckpt_last|ckpt_robust]')
        model.load(os.path.join(args['output_path']['models'], args['pretrained']))
    if args['training_type'] == 'vanilla':
        print('Vanilla training.')
        train_vanilla(model, train_loader, test_loader, args, device=device)
    elif args['training_type'] == 'stochastic':
        print('Stochastic training.')
        train_stochastic(model, train_loader, test_loader, args, device=device)
    elif args['training_type'] == 'stochastic+adversarial':
        print('Adversarial stochastic training.')
        train_stochastic_adversarial(model, train_loader, test_loader, args, device=device)
    else:
        raise NotImplementedError(
            'Training "{}" not implemented. Supported: [vanilla|stochastic|stochastic+adversarial].'.format(
                args['training_type']))
    print('Finished training.')


def test(args, device):
    print(args)
    model = model_factory(
        args['dataset'], args['training_type'], args['var_type'], args['feature_dim'], args['num_classes'])
    model.to(device)
    # model.load(os.path.join(args['output_path']['models'], 'ckpt_best'))
    model.load(os.path.join(args['output_path']['models'], 'ckpt_last'))
    model.eval()
    test_loader = get_data_loader(args['dataset'], args['batch_size'], False, shuffle=False, drop_last=False)
    attack_names = ['FGSM', 'PGD']  # 'BIM', 'C&W', 'Few-Pixel'
    print('Adversarial testing.')
    for idx, attack in enumerate(attack_names):
        print('Attack: {}'.format(attack))
        if attack == 'Few-Pixel':
            if args['dataset'] == 'cifar10':
                preproc = {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]}
            else:
                raise NotImplementedError('Only CIFAR-10 supported for the one-pixel attack.')
            one_pixel_attack(
                model, test_loader, preproc, device, pixels=1, targeted=False, maxiter=1000, popsize=400, verbose=False)
        else:
            eps_names = attack_to_dataset_config[attack][args['dataset']]['eps_names']
            eps_values = attack_to_dataset_config[attack][args['dataset']]['eps_values']
            robust_accuracy = test_attack(model, test_loader, attack, eps_values, args, device)
            for eps_name, eps_value, accuracy in zip(eps_names, eps_values, robust_accuracy):
                print('Attack Strength: {}, Accuracy: {:.3f}'.format(eps_name, accuracy.item()))
    print('Finished testing.')


def main(mode, args):
    if args['device'] is not None:
        device = args['device']
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if mode == 'train':
        train(args, device)
    elif mode == 'test':
        test(args, device)
    else:
        train(args, device)
        test(args, device)


if __name__ == '__main__':
    try:
        mode, args = parse_args()
    except ValueError:
        print('Invalid mode. Usage: python run.py <mode[train|test|train+test]> <config. file>')
    except IndexError:
        print('Path to configuration file missing. Usage: python run.py <mode[train|test|train+test]> <config. file>')
        sys.exit()
    except FileNotFoundError:
        print('Incorrect path to configuration file. File not found.')
        sys.exit()
    except json.JSONDecodeError:
        print('Configuration file is an invalid JSON.')
        sys.exit()
    main(mode, args)
