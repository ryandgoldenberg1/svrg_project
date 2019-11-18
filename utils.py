import torch
from torchvision import datasets, transforms


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


def calculate_full_gradient_norm(model, data_loader, loss_fn, device):
    grad = calculate_full_gradient(model=model, data_loader=data_loader, loss_fn=loss_fn, device=device)
    return torch.norm(grad, 2).item()


def get_dataset(dataset, root, download):
    kwargs = {'root': root, 'download': download, 'transform': transforms.ToTensor()}
    dataset_to_fn = {'MNIST': datasets.MNIST, 'CIFAR10': datasets.CIFAR10, 'STL10': datasets.STL10}
    assert dataset in dataset_to_fn, 'Unrecognized dataset: {}'.format(dataset)
    return dataset_to_fn[dataset](**kwargs)
