import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

examples = next(iter(train_loader))

images, labels = examples

x0 = images[0:1]
label = labels[0]

print("Shape immagine:", x0.shape)
print("Label:", label.item())
