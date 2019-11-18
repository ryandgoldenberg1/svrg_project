import copy
import json
import random
import time

import torch

import utils


class SVRGTrainer:
    def __init__(self, create_model, loss_fn):
        self.create_model = create_model
        self.loss_fn = loss_fn

    def train(self, *, train_loader, test_loader, num_warmup_epochs, num_outer_epochs, num_inner_epochs,
              inner_epoch_fraction, warmup_learning_rate, learning_rate, device, weight_decay, choose_random_iterate,
              **kwargs):
        print('SVRGTrainer Hyperparameters:', json.dumps({
            'num_warmup_epochs': num_warmup_epochs,
            'num_outer_epochs': num_outer_epochs,
            'num_inner_epochs': num_inner_epochs,
            'inner_epoch_fraction': inner_epoch_fraction,
            'warmup_learning_rate': warmup_learning_rate,
            'learning_rate': learning_rate,
            'device': device,
            'weight_decay': weight_decay,
            'choose_random_iterate': choose_random_iterate
        }, indent=2))
        print('Unused kwargs:', kwargs)

        device = torch.device(device)
        metrics = []

        model = self.create_model().to(device)
        target_model = self.create_model().to(device)
        print(model)

        # Perform several epochs of SGD as initialization for SVRG
        warmup_optimizer = torch.optim.SGD(
            target_model.parameters(), lr=warmup_learning_rate, weight_decay=weight_decay)
        for warmup_epoch in range(1, num_warmup_epochs + 1):
            warmup_loss = 0
            epoch_start = time.time()
            for batch in train_loader:
                data, label = (x.to(device) for x in batch)
                warmup_optimizer.zero_grad()
                prediction = target_model(data.to(device))
                loss = self.loss_fn(prediction, label.to(device))
                loss.backward()
                warmup_optimizer.step()
                warmup_loss += loss.item() * len(data)
            avg_warmup_loss = warmup_loss / len(train_loader.dataset)
            model_grad_norm = utils.calculate_full_gradient_norm(
                model=target_model, data_loader=train_loader, loss_fn=self.loss_fn, device=device)
            test_error = utils.calculate_error(model=target_model, data_loader=test_loader, device=device)
            elapsed_time = time.time() - epoch_start
            ex_per_sec = len(train_loader.dataset) / elapsed_time
            metrics.append({'warmup_epoch': warmup_epoch,
                            'train_loss': avg_warmup_loss,
                            'grad_norm': model_grad_norm,
                            'test_error': test_error})
            print('[Warmup {}/{}] loss: {:.04f}, grad_norm: {:.02f}, test_error: {:.04f}, (1k) ex/s: {:.02f}'.format(
                warmup_epoch, num_warmup_epochs, avg_warmup_loss, model_grad_norm, test_error, ex_per_sec / 1000))

        for epoch in range(1, num_outer_epochs + 1):
            # Find full target gradient
            mu = utils.calculate_full_gradient(
                model=target_model,
                data_loader=train_loader,
                loss_fn=self.loss_fn,
                device=device
            )

            # Initialize model to target model
            model.load_state_dict(copy.deepcopy(target_model.state_dict()))

            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            model_state_dicts = []
            inner_batches = len(train_loader)
            if inner_epoch_fraction is not None:
                inner_batches = int(len(train_loader) * inner_epoch_fraction)
            for sub_epoch in range(1, num_inner_epochs + 1):
                train_loss = 0
                examples_seen = 0
                epoch_start = time.time()
                for batch_idx, batch in enumerate(train_loader):

                    data, label = (x.to(device) for x in batch)
                    optimizer.zero_grad()

                    target_model.zero_grad()
                    target_model_out = target_model(data)
                    target_model_loss = self.loss_fn(target_model_out, label)
                    target_model_loss.backward()
                    target_model_grad = torch.cat(
                        [x.grad.view(-1) for x in target_model.parameters()]).detach()

                    model_weights = torch.cat(
                        [x.view(-1) for x in model.parameters()])
                    model_out = model(data)
                    model_loss = self.loss_fn(model_out, label)

                    # Use SGD on auxiliary loss function
                    # See the SVRG paper section 2 for details
                    aux_loss = model_loss - \
                        torch.dot((target_model_grad - mu).detach(),
                                  model_weights)
                    aux_loss.backward()
                    optimizer.step()

                    train_loss += model_loss.item() * len(data)
                    examples_seen += len(data)
                    copy_state_dict = copy.deepcopy(model.state_dict())
                    # Copy model parameters to CPU first to prevent GPU overflow
                    for k, v in copy_state_dict.items():
                        copy_state_dict[k] = v.cpu()
                    model_state_dicts.append(copy_state_dict)

                    batch_num = batch_idx + 1
                    if batch_num >= inner_batches:
                        break
                avg_train_loss = train_loss / examples_seen
                model_grad_norm = utils.calculate_full_gradient_norm(
                    model=model, data_loader=train_loader, loss_fn=self.loss_fn, device=device)
                test_error = utils.calculate_error(model=model, data_loader=test_loader, device=device)
                elapsed_time = time.time() - epoch_start
                ex_per_sec = len(train_loader.dataset) / elapsed_time
                metrics.append({'outer_epoch': epoch,
                                'inner_epoch': sub_epoch,
                                'train_loss': avg_train_loss,
                                'grad_norm': model_grad_norm,
                                'test_error': test_error})
                print('[Outer {}/{}, Inner {}/{}] loss: {:.04f}, grad_norm: {:.02f}, test_error: {:.04f}, (1k) ex/s: {:.02f}'.format(
                    epoch, num_outer_epochs, sub_epoch, num_inner_epochs, avg_train_loss, model_grad_norm, test_error,
                    ex_per_sec / 1000))  # noqa

            if choose_random_iterate:
                new_target_state_dict = random.choice(model_state_dicts)
            else:
                new_target_state_dict = model_state_dicts[-1]
            target_model.load_state_dict(new_target_state_dict)
        return metrics
