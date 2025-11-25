import torch
from torchvision import datasets, transforms

def get_mnist_dataloader(batch_size=16, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader