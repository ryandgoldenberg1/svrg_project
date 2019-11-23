import torch

class Rosenbrock:
    def __init__(self):
        pass
    def __call__(self, x, y):
        return torch.mean((1 - x) ** 2 + 100 * (y - x ** 2) ** 2)

class BenchMarkFunction(torch.nn.Module):
    def __init__(self, num_data, batch_size):
        super(BenchMarkFunction, self).__init__()
        self.num_data = num_data
        self.x = torch.nn.Parameter(torch.randn(num_data, 1))
        self.y = torch.nn.Parameter(torch.randn(num_data, 1))
        self.start_idx = 0
        self.end_idx = 0
        self.batch_size = batch_size
        self.num_iters = num_data // batch_size
    def forward(self):
        self.end_idx = self.start_idx + self.batch_size
        if self.end_idx >= self.num_data:
            x = self.x[self.start_idx:]
            y = self.y[self.start_idx:]
            self.start_idx = 0
            return x, y
        else:
            x = self.x[self.start_idx:self.end_idx]
            y = self.y[self.start_idx:self.end_idx]
            self.start_idx = self.end_idx
            return x, y
def calculate_full_gradient_benchfunc(model, loss_fn, device):
    model.zero_grad()
    for batch in range(model.num_iters):
        x, y = model()
        x, y = x.to(device), y.to(device)
        loss = loss_fn(x,y)
        loss *= len(x) / model.num_data
        loss.backward()
    gradient = torch.cat([p.grad.view(-1) for p in model.parameters()]).detach()
    model.zero_grad()
    return gradient


def calculate_full_gradient_norm_benchfunc(model, loss_fn, device):
    grad = calculate_full_gradient_benchfunc(model=model, loss_fn=loss_fn, device=device)
    return torch.norm(grad, 2).item()

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
