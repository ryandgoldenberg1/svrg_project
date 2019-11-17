import copy
import random
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class SDGTrainer:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train(self, train_loader, num_epochs):
        epoch_avg_losses = []
        for epoch in range(1, num_epochs+1):
            train_loss = 0
            for batch in train_loader:
                # Training Step
                data, label = batch
                self.optimizer.zero_grad()
                prediction = self.model(data)
                loss = self.loss_fn(prediction, label)
                loss.backward()
                self.optimizer.step()
                # Update Statistics
                train_loss += loss.item() * data.shape[0]
            avg_train_loss = train_loss / len(train_loader.dataset)
            epoch_avg_losses.append(avg_train_loss)
            print('[Epoch {}] train_loss: {:.04f}'.format(epoch, avg_train_loss))
        return epoch_avg_losses


def create_mlp(layer_sizes):
    layers = [nn.Flatten()]
    for i in range(1, len(layer_sizes)):
        in_size = layer_sizes[i - 1]
        out_size = layer_sizes[i]
        layers.append(nn.Linear(in_size, out_size))
        layers.append(nn.ReLU())
    layers.pop()
    return nn.Sequential(*layers)


if __name__ == '__main__':
    dataset_size = None
    # dataset_size = 600
    batch_size = 1
    num_epochs = 100
    # num_epochs = 10
    learning_rate = 0.001
    weight_decay = 0.0001

    train_ds = datasets.MNIST('~/datasets/pytorch', transform=transforms.ToTensor(), download=True)
    if dataset_size is not None:
        train_ds = torch.utils.data.dataset.Subset(train_ds, indices=list(range(dataset_size)))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    print('dataset_size:', len(train_loader.dataset))

    model = create_mlp(layer_sizes=[784, 10])
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    trainer = SDGTrainer(model=model, loss_fn=loss_fn, optimizer=optimizer)
    losses = trainer.train(train_loader, num_epochs=num_epochs)
    plt.plot(losses)
    plt.show()
