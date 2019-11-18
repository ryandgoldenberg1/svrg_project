import argparse
import copy
import functools
import json
import random
import sys
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms

from sgd import SGDTrainer
from svrg import SVRGTrainer


def create_mlp(layer_sizes):
    layers = [nn.Flatten()]
    for i in range(1, len(layer_sizes)):
        in_size = layer_sizes[i - 1]
        out_size = layer_sizes[i]
        layers.append(nn.Linear(in_size, out_size))
        layers.append(nn.ReLU())
    layers.pop()
    return nn.Sequential(*layers)


def get_dataset(dataset, root, download):
    kwargs = {'root': root, 'download': download, 'transform': transforms.ToTensor()}
    dataset_to_fn = {'MNIST': datasets.MNIST, 'CIFAR10': datasets.CIFAR10, 'STL10': datasets.STL10}
    assert dataset in dataset_to_fn, 'Unrecognized dataset: {}'.format(dataset)
    return dataset_to_fn[dataset](**kwargs)


def create_trainer(args):
    device = torch.device(args.device)
    loss_fn = nn.CrossEntropyLoss()

    if args.optimizer == 'SGD':
        model = create_mlp(layer_sizes=args.layer_sizes).to(device)
        print(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        return SGDTrainer(model=model, loss_fn=loss_fn, optimizer=optimizer)
    elif args.optimizer == 'SVRG':
        create_model = functools.partial(create_mlp, layer_sizes=args.layer_sizes)
        return SVRGTrainer(create_model=create_model, loss_fn=loss_fn)
    else:
        raise ValueError('Unrecognized optimizer: {}'.format(args.optimizer))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('--optimizer', choices=['SGD', 'SVRG'], required=True)
    parser.add_argument('--run_name', default='train')
    parser.add_argument('--output_path')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    # Dataset Args
    group = parser.add_argument_group('dataset arguments')
    group.add_argument('--dataset', default='MNIST', choices=['MNIST', 'CIFAR10', 'STL10'])
    group.add_argument('--dataset_root', default='~/datasets/pytorch')
    group.add_argument('--dataset_size', type=int)
    group.add_argument('--download', default=False, action='store_true')
    # Common Hyperparameters
    group = parser.add_argument_group('common hyperparameters')
    group.add_argument('--layer_sizes', type=int, nargs='+', default=[784, 10])
    group.add_argument('--batch_size', type=int, default=1)
    group.add_argument('--learning_rate', type=float, default=0.025)
    group.add_argument('--weight_decay', type=float, default=0.0001)
    # SVRG Specific Arguments
    group = parser.add_argument_group('SVRG hyperparameters')
    group.add_argument('--warmup_learning_rate', type=float, default=0.01)
    group.add_argument('--num_warmup_epochs', type=int, default=10)
    group.add_argument('--num_outer_epochs', type=int, default=100)
    group.add_argument('--num_inner_epochs', type=int, default=5)
    group.add_argument('--inner_epoch_fraction', type=float)
    group.add_argument('--choose_random_iterate', default=False, action='store_true')
    # SGD Specific Arguments
    group = parser.add_argument_group('SGD hyperparams')
    group.add_argument('--num_epochs', type=int, default=10)
    args = parser.parse_args()
    print('Command Line Args:', json.dumps(args.__dict__, indent=2))

    if args.seed is not None:
        print('Using seed:', args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_ds = get_dataset(dataset=args.dataset, root=args.dataset_root, download=args.download)
    print(train_ds)
    if args.dataset_size is not None and len(train_ds) > args.dataset_size:
        print('Limiting dataset size to:', args.dataset_size)
        train_ds = torch.utils.data.dataset.Subset(train_ds, indices=list(range(args.dataset_size)))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    trainer = create_trainer(args)
    print(trainer)

    metrics = trainer.train(train_loader=train_loader, **args.__dict__)
    output = {'argv': sys.argv, 'args': args.__dict__, 'metrics': metrics}
    if args.output_path is not None:
        with open(args.output_path, 'w') as f:
            json.dump(output, f)
        print('Wrote output to:', args.output_path)


if __name__ == '__main__':
    main()
