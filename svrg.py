import copy
import random
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms


class SDGTrainer:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train(self, train_loader, num_epochs):
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
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            print('[Epoch {}] train_loss: {:.02f}'.format(epoch, avg_train_loss))


class SVRGTrainer:
    def __init__(self, create_model, loss_fn):
        self.create_model = create_model
        self.loss_fn = loss_fn

    def train(self, train_loader, num_warmup_epochs, num_epochs, num_sub_epochs, learning_rate, weight_decay):
        model = self.create_model()
        target_model = self.create_model()

        warmup_optimizer = torch.optim.SGD(target_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for warmup_epoch in range(1, num_warmup_epochs+1):
            warmup_loss = 0
            for batch in train_loader:
                data, label = batch
                warmup_optimizer.zero_grad()
                prediction = target_model(data)
                loss = self.loss_fn(prediction, label)
                loss.backward()
                warmup_optimizer.step()
                warmup_loss += loss.item()
            avg_warmup_loss = warmup_loss / len(train_loader)
            print('[Warmup {}/{}] loss: {:.02f}'.format(warmup_epoch, num_warmup_epochs, avg_warmup_loss))

        for epoch in range(1, num_epochs+1):
            # Find full target gradient
            target_model.zero_grad()
            for batch in train_loader:
                data, label = batch
                prediction = target_model(data)
                loss = self.loss_fn(prediction, label)
                # The loss function averages over the batch, len(data).
                # In order to average over all examples, we need to scale it.
                loss *= len(data) / len(train_loader.dataset)
                loss.backward()
            mu = torch.cat([ x.grad.view(-1) for x in target_model.parameters() ]).detach()
            target_model.zero_grad()

            # Initialize model to target model
            model.load_state_dict(copy.deepcopy(target_model.state_dict()))

            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            model_state_dicts = []
            for sub_epoch in range(1, num_sub_epochs+1):
                train_loss = 0
                for batch in train_loader:
                    data, label = batch
                    optimizer.zero_grad()

                    target_model.zero_grad()
                    target_model_out = target_model(data)
                    target_model_loss = self.loss_fn(target_model_out, label)
                    target_model_loss.backward()
                    target_model_grad = torch.cat([ x.grad.view(-1) for x in target_model.parameters() ]).detach()

                    model_weights = torch.cat([ x.view(-1) for x in model.parameters() ])
                    model_out = model(data)
                    model_loss = self.loss_fn(model_out, label)

                    aux_loss = model_loss - torch.dot((target_model_grad - mu).detach(), model_weights)
                    aux_loss.backward()
                    optimizer.step()

                    train_loss += model_loss.item()
                    model_state_dicts.append(copy.deepcopy(model.state_dict()))
                avg_train_loss = train_loss / len(train_loader)
                print('[Outer {}/{}, Inner {}/{}] loss: {:.02f}'.format(epoch, num_epochs, sub_epoch, num_sub_epochs, avg_train_loss))

            new_target_state_dict = random.choice(model_state_dicts)
            target_model.load_state_dict(new_target_state_dict)


def create_mlp():
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 10))



if __name__ == '__main__':
    dataset_size = 600
    batch_size = 1
    num_epochs = 10
    num_sub_epochs = 2
    num_warmup_epochs = 3
    learning_rate = 0.001
    weight_decay = 0.001

    train_ds = datasets.MNIST('~/datasets/pytorch', transform=transforms.ToTensor())
    train_ds = torch.utils.data.dataset.Subset(train_ds, indices=list(range(dataset_size)))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = create_mlp()
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # trainer = SDGTrainer(model=model, loss_fn=loss_fn, optimizer=optimizer)
    # trainer.train(train_loader, num_epochs=num_epochs*10)
    trainer = SVRGTrainer(create_model=create_mlp, loss_fn=loss_fn)
    trainer.train(train_loader, num_epochs=num_epochs, num_sub_epochs=num_sub_epochs,
                  learning_rate=learning_rate, num_warmup_epochs=num_warmup_epochs, weight_decay=weight_decay)
