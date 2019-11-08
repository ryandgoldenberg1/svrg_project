import argparse
import copy
import functools
import json
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms


class SVRGTrainer:
    def __init__(self, create_model, loss_fn):
        self.create_model = create_model
        self.loss_fn = loss_fn

    def train(self, train_loader, num_warmup_epochs, num_outer_epochs, num_inner_epochs, learning_rate, device):
        metrics = []
        grad_epoch = 0

        model = self.create_model().to(device)
        target_model = self.create_model().to(device)
        print(model)

        # Perform several epochs of SGD as initialization for SVRG
        warmup_optimizer = torch.optim.SGD(target_model.parameters(), lr=learning_rate)
        for warmup_epoch in range(1, num_warmup_epochs+1):
            warmup_loss = 0
            for batch in train_loader:
                data, label = (x.to(device) for x in batch)
                warmup_optimizer.zero_grad()
                prediction = target_model(data.to(device))
                loss = self.loss_fn(prediction, label.to(device))
                loss.backward()
                warmup_optimizer.step()
                warmup_loss += loss.item() * len(data)
            avg_warmup_loss = warmup_loss / len(train_loader.dataset)
            grad_epoch += 1
            metrics.append({'grad_epoch': grad_epoch, 'train_loss': avg_warmup_loss})
            print('[Warmup {}/{}] # grad/n: {}, loss: {:.02f}'.format(
                warmup_epoch, num_warmup_epochs, grad_epoch, avg_warmup_loss))

        for epoch in range(1, num_outer_epochs+1):
            # Find full target gradient
            target_model.zero_grad()
            for batch in train_loader:
                data, label = (x.to(device) for x in batch)
                prediction = target_model(data)
                loss = self.loss_fn(prediction, label)
                # The loss function averages over the batch, len(data).
                # In order to average over all examples, we need to scale it.
                loss *= len(data) / len(train_loader.dataset)
                loss.backward()
            mu = torch.cat([ x.grad.view(-1) for x in target_model.parameters() ]).detach()
            target_model.zero_grad()
            grad_epoch += 1

            # Initialize model to target model
            model.load_state_dict(copy.deepcopy(target_model.state_dict()))

            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            model_state_dicts = []
            for sub_epoch in range(1, num_inner_epochs+1):
                train_loss = 0
                step = 0
                for batch in train_loader:
                    data, label = (x.to(device) for x in batch)
                    optimizer.zero_grad()

                    target_model.zero_grad()
                    target_model_out = target_model(data)
                    target_model_loss = self.loss_fn(target_model_out, label)
                    target_model_loss.backward()
                    target_model_grad = torch.cat([ x.grad.view(-1) for x in target_model.parameters() ]).detach()

                    model_weights = torch.cat([ x.view(-1) for x in model.parameters() ])
                    model_out = model(data)
                    model_loss = self.loss_fn(model_out, label)

                    # Use SGD on auxiliary loss function
                    # See the SVRG paper section 2 for details
                    aux_loss = model_loss - torch.dot((target_model_grad - mu).detach(), model_weights)
                    aux_loss.backward()
                    optimizer.step()

                    train_loss += model_loss.item() * len(data)
                    model_state_dicts.append(copy.deepcopy(model.state_dict()))
                avg_train_loss = train_loss / len(train_loader.dataset)
                grad_epoch += 1
                metrics.append({'grad_epoch': grad_epoch, 'train_loss': avg_train_loss})
                print('[Outer {}/{}, Inner {}/{}] # grad/n: {}, loss: {:.03f}'.format(epoch, num_outer_epochs, sub_epoch, num_inner_epochs, grad_epoch, avg_train_loss))
            new_target_state_dict = random.choice(model_state_dicts)
            target_model.load_state_dict(new_target_state_dict)
        return metrics


def create_mlp(layer_sizes):
    layers = [nn.Flatten()]
    for i in range(1, len(layer_sizes)):
        in_size = layer_sizes[i-1]
        out_size = layer_sizes[i]
        layers.append(nn.Linear(in_size, out_size))
        layers.append(nn.ReLU())
    layers.pop()
    return nn.Sequential(*layers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('--dataset_path', default='~/datasets/pytorch')
    parser.add_argument('--max_dataset_size', type=int)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--layer_sizes', type=int, nargs='+', default=[784, 100, 10])
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--num_warmup_epochs', type=int, default=10)
    parser.add_argument('--num_outer_epochs', type=int, default=100)
    parser.add_argument('--num_inner_epochs', type=int, default=5)
    parser.add_argument('--metrics_path')
    parser.add_argument('--plot', default=False, action='store_true')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    if args.seed is not None:
        print('Using seed:', args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_ds = datasets.MNIST(args.dataset_path, transform=transforms.ToTensor())
    if args.max_dataset_size is not None and len(train_ds) > args.max_dataset_size:
        print('Limiting dataset size to:', args.max_dataset_size)
        train_ds = torch.utils.data.dataset.Subset(train_ds, indices=list(range(args.max_dataset_size)))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    create_model = functools.partial(create_mlp, layer_sizes=args.layer_sizes)
    trainer = SVRGTrainer(create_model=create_model, loss_fn=loss_fn)
    metrics = trainer.train(
        train_loader=train_loader,
        num_warmup_epochs=args.num_warmup_epochs,
        num_outer_epochs=args.num_outer_epochs,
        num_inner_epochs=args.num_inner_epochs,
        learning_rate=args.learning_rate,
        device=torch.device(args.device))

    if args.metrics_path is not None:
        with open(args.metrics_path, 'w') as f:
            json.dump(metrics, f)
        print('Wrote metrics to:', args.metrics_path)

    if args.plot:
        x = [el['grad_epoch'] for el in metrics]
        y = [el['train_loss'] for el in metrics]
        plt.plot(x, y)
        plt.show()


if __name__ == '__main__':
    main()
