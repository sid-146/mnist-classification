from models import MNIST_CNN

import torch
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

# training dataset
train_ds = datasets.MNIST(
    "../data",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]),
)

batch_size = 100
validation_split = 0.1
shuffle_dataset=True

dataset_size = len(train_ds)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
random_seed = 2

if shuffle_dataset:
    