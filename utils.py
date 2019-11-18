import torch


def calculate_full_gradient(model, data_loader, loss_fn, device):
    model.zero_grad()
    for batch in data_loader:
        data, label = (x.to(device) for x in batch)
        prediction = model(data)
        loss = loss_fn(prediction, label)
        loss *= len(data) / len(data_loader.dataset)
        loss.backward()
    gradient = torch.cat([x.grad.view(-1) for x in model.parameters()]).detach()
    model.zero_grad()
    return gradient


def calculate_full_gradient_norm(model, data_loader, loss_fn, device):
    grad = calculate_full_gradient(model=model, data_loader=data_loader, loss_fn=loss_fn, device=device)
    return torch.norm(grad, 2).item()


def calculate_error(model, data_loader, device):
    num_incorrect = 0
    with torch.no_grad():
        for batch in data_loader:
            data, label = (x.to(device) for x in batch)
            prediction = model(data)
            batch_size = len(data)
            assert label.shape == (batch_size,), 'Expected label shape {}. Got {}'.format((batch_size,), label.shape)
            assert prediction.shape[0] == batch_size and len(prediction.shape) == 2
            prediction = prediction.argmax(dim=1)
            assert prediction.shape == (batch_size,)
            num_incorrect += (prediction != label).sum().item()
    return num_incorrect / len(data_loader.dataset)
