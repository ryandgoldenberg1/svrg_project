import copy
import json
import random
import time

import torch

import utils
class SVRGTrainer_benchfunc:
    def __init__(self, create_model, loss_fn):
        self.create_model = create_model
        self.loss_fn = loss_fn

    def train(self, *, num_warmup_epochs, num_outer_epochs, num_inner_epochs,
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

        model = self.create_model.to(device)
        target_model = self.create_model.to(device)
        print(model)
        # Perform several epochs of SGD as initialization for SVRG
        warmup_optimizer = torch.optim.SGD(
            target_model.parameters(), lr=warmup_learning_rate, weight_decay=weight_decay)
        for warmup_epoch in range(1, num_warmup_epochs + 1):
            warmup_loss = 0
            epoch_start = time.time()
            for batch in range(model.num_iters):
                x, y = target_model()
                x, y = x.to(device), y.to(device)
                warmup_optimizer.zero_grad()
                loss = self.loss_fn(x, y)
                loss.backward()
                warmup_optimizer.step()
                warmup_loss += loss.item() * len(x)
            avg_warmup_loss = warmup_loss / model.num_data
            model_grad_norm = utils.calculate_full_gradient_norm_benchfunc(model=target_model, loss_fn=self.loss_fn, device=device)
            elapsed_time = time.time() - epoch_start
            ex_per_sec = model.num_data / elapsed_time
            metrics.append({'warmup_epoch': warmup_epoch,
                            'train_loss': avg_warmup_loss,
                            'grad_norm': model_grad_norm})
            print('[Warmup {}/{}] loss: {:.04f}, grad_norm: {:.02f}, (1k) ex/s: {:.02f}'.format(
                warmup_epoch, num_warmup_epochs, avg_warmup_loss, model_grad_norm, ex_per_sec / 1000))

        for epoch in range(1, num_outer_epochs + 1):
            # Find full target gradient
            mu = utils.calculate_full_gradient_norm_benchfunc(model=target_model, loss_fn=self.loss_fn, device=device)
            #print("model state dict before")
            #print(model.state_dict())
            # Initialize model to target model
            model.load_state_dict(copy.deepcopy(target_model.state_dict()))

            #print("model state dict after")
            #print(model.state_dict())
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            model_state_dicts = []
            inner_batches = model.num_data
            if inner_epoch_fraction is not None:
                inner_batches = int(len(train_loader) * inner_epoch_fraction)
            for sub_epoch in range(1, num_inner_epochs + 1):
                train_loss = 0
                examples_seen = 0
                epoch_start = time.time()
                for batch_idx in range(model.num_iters):

                    x_trg, y_trg = target_model()
 
                    x_trg, y_trg = x_trg.to(device), y_trg.to(device)
                    optimizer.zero_grad()

                    target_model.zero_grad()
                    target_loss = self.loss_fn(x_trg, y_trg)
                    print("Targeet loss", target_loss)
                    target_loss.backward()
                    print(target_model.x.grad)
                    target_model_grad = torch.cat(
                        [p.grad.view(-1) for p in target_model.parameters()]).detach()
                    # print("Target model grad:")
                    # print(target_model_grad)
                    model_weights = torch.cat(
                        [p.view(-1) for p in model.parameters()])
                    model_loss =  self.loss_fn(x_trg, y_trg)

                    # Use SGD on auxiliary loss function
                    # See the SVRG paper section 2 for details
                    aux_loss = model_loss - torch.dot((target_model_grad - mu).detach(), model_weights)
                    aux_loss.backward()
                    optimizer.step()

                    train_loss += model_loss.item() * len(x)
                    examples_seen += len(x)
                    copy_state_dict = copy.deepcopy(model.state_dict())
                    # print("Copy state_dict")
                    # print(copy_state_dict)
                    # Copy model parameters to CPU first to prevent GPU overflow
                    for k, v in copy_state_dict.items():
                        copy_state_dict[k] = v.cpu()
                    model_state_dicts.append(copy_state_dict)

                    batch_num = batch_idx + 1
                    if batch_num >= inner_batches:
                        break
                avg_train_loss = train_loss / examples_seen
                model_grad_norm = utils.calculate_full_gradient_norm_benchfunc(model=model, loss_fn=self.loss_fn, device=device)
                elapsed_time = time.time() - epoch_start
                ex_per_sec = model.num_data / elapsed_time
                metrics.append({'outer_epoch': epoch,
                                'inner_epoch': sub_epoch,
                                'train_loss': avg_train_loss,
                                'grad_norm': model_grad_norm})
                print('[Outer {}/{}, Inner {}/{}] loss: {:.04f}, grad_norm: {:.02f}, (1k) ex/s: {:.02f}'.format(
                    epoch, num_outer_epochs, sub_epoch, num_inner_epochs, avg_train_loss, model_grad_norm,
                    ex_per_sec / 1000))  # noqa

            if choose_random_iterate:
                new_target_state_dict = random.choice(model_state_dicts)
            else:
                new_target_state_dict = model_state_dicts[-1]
            target_model.load_state_dict(new_target_state_dict)
        return metrics
