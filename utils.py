import torch


def calculate_full_gradient(model, data_loader, loss_fn, device):
    model.zero_grad()
    for batch in data_loader:
        data, label = (x.to(device) for x in batch)
        prediction = model(data)
        loss = loss_fn(prediction, label)
        loss *= len(data) / len(data_loader.dataset)
        loss.backward()
    gradient = torch.cat([ x.grad.view(-1) for x in model.parameters() ]).detach()
    model.zero_grad()
    return gradient
