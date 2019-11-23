import json

import torch

import utils


class SGDTrainer:
    """Class for training models using Stochastic Gradient Descent.

    model: torch.nn.Module Pytorch neural network
    loss_fn: Function of the form (predictions, labels) -> loss
    optimizer: torch.optim.Optimizer should be the SGD optimizer
    """
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train(self, *, train_loader, test_loader, num_epochs, device, **kwargs):
        """Executes training for the model over the given dataset and hyperparameters.

        Inputs:
        train_loader: torch.utils.data.DataLoader data loader for the training dataset
        test_loader: torch.utils.data.DataLoader data loader for the test dataset
        num_epochs: the number of epochs to train for
        device: string denoting the device to run on. "cuda" or "cpu" are expected.
        kwargs: any additional keyword arguments will be excepted but ignored.

        Returns:
        metrics: a list of dictionaries containing information about the run, including the training loss,
            gradient norm, and test error for each epoch.
        """
        print('SGDTrainer Hyperparameters:', json.dumps({'num_epochs': num_epochs, 'device': device}, indent=2))
        print('Unused kwargs:', kwargs)

        device = torch.device(device)
        metrics = []
        for epoch in range(1, num_epochs + 1):
            train_loss = 0
            for batch in train_loader:
                # Training Step
                data, label = (x.to(device) for x in batch)
                self.optimizer.zero_grad()
                prediction = self.model(data)
                loss = self.loss_fn(prediction, label)
                loss.backward()
                self.optimizer.step()
                # Update Statistics
                train_loss += loss.item() * data.shape[0]
            avg_train_loss = train_loss / len(train_loader.dataset)
            model_grad_norm = utils.calculate_full_gradient_norm(
                model=self.model, data_loader=train_loader, loss_fn=self.loss_fn, device=device)
            test_error = utils.calculate_error(model=self.model, data_loader=test_loader, device=device)
            metrics.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'grad_norm': model_grad_norm,
                'test_error': test_error
            })
            print('[Epoch {}] train_loss: {:.04f}, grad_norm: {:.02f}, test_error: {:.04f}'.format(
                epoch, avg_train_loss, model_grad_norm, test_error))
        return metrics
