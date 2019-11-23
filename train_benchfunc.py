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

from sgd_benchfunc import SGDTrainer_benchfunc
from svrg_benchfunc import SVRGTrainer_benchfunc
from utils import BenchMarkFunction, Rosenbrock

def create_trainer(args):
    device = torch.device(args.device)
    if args.function_name == 'Rosenbrock':
        loss_fn = Rosenbrock()
    elif args.function_name == 'eggholder':
        pass
    else:
        raise ValueError('Unrecognized benchmark function: {}'.format(args.benchfunc))

    if args.optimizer == 'SGD':
        model = BenchMarkFunction(num_data=args.num_data_benchfunc, batch_size=args.batch_size).to(device)
        print(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        return SGDTrainer_benchfunc(model=model, loss_fn=loss_fn, optimizer=optimizer)
    elif args.optimizer == 'SVRG':
        create_model = BenchMarkFunction(num_data=args.num_data_benchfunc, batch_size=args.batch_size).to(device)
        return SVRGTrainer_benchfunc(create_model=create_model, loss_fn=loss_fn)
    else:
        raise ValueError('Unrecognized optimizer: {}'.format(args.optimizer))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('--optimizer', choices=['SGD', 'SVRG'], required=True)
    parser.add_argument('--run_name', default='train')
    parser.add_argument('--output_path')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    
    group = parser.add_argument_group('benchmark functions') 
    group.add_argument('--function_name', choices=['Rosenbrock', 'Eggholder'], required=True)  
    group.add_argument('--num_data_benchfunc', type=int, default=500)
    # Common Hyperparameters
    group = parser.add_argument_group('common hyperparameters')
    group.add_argument('--layer_sizes', type=int, nargs='+', default=[784, 10])
    group.add_argument('--batch_size', type=int, default=1)
    group.add_argument('--test_batch_size', type=int, default=256)
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

    trainer = create_trainer(args)
    print(trainer)

    metrics = trainer.train(**args.__dict__)
    output = {'argv': sys.argv, 'args': args.__dict__, 'metrics': metrics}
    if args.output_path is not None:
        with open(args.output_path, 'w') as f:
            json.dump(output, f)
        print('Wrote output to:', args.output_path)


if __name__ == '__main__':
    main()
