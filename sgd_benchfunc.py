import json

import torch

import utils


class SGDTrainer_benchfunc:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train(self, *, num_epochs, device, **kwargs):
        print('SGDTrainer Hyperparameters:', json.dumps({'num_epochs': num_epochs, 'device': device}, indent=2))
        print('Unused kwargs:', kwargs)

        device = torch.device(device)
        metrics = []
        for epoch in range(1, num_epochs + 1):
            train_loss = 0
                # Training Step
            for batch in range(self.model.num_iters):
                self.optimizer.zero_grad()
                x, y = self.model()
                x, y = x.to(device), y.to(device)
                loss = self.loss_fn(x, y)
                loss.backward()
                self.optimizer.step()
                # Update Statistics
                train_loss += loss.item() * len(x)

            avg_train_loss = train_loss /self.model.num_data
            model_grad_norm = utils.calculate_full_gradient_norm_benchfunc(model=self.model,\
            loss_fn=self.loss_fn, device=device)
            metrics.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'grad_norm': model_grad_norm,
            })
            print('[Epoch {}] train_loss: {:.04f}, grad_norm: {:.02f}'.format(
                epoch, avg_train_loss, model_grad_norm))
        return metrics
