import torch.nn as nn
import torch.nn.functional as F


class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()

        # first Conv1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=1)  # o/p 26*26*32
        self.maxPool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # o/p 13*13*32
        self.batch1 = nn.BatchNorm2d(32)

        # second Conv2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1)  # o/p 11*11*64
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # o/p 5*5*64
        self.batch2 = nn.BatchNorm2d(64)

        # Flatten Layers
        self.fc1 = nn.Linear(5 * 5 * 64, 2000)
        self.fc2 = nn.Linear(2000, 1000)
        self.fc3 = nn.Linear(1000, 500)
        self.fc4 = nn.Linear(500, 100)
        self.fc5 = nn.Linear(100, 10)

    def forward(self, x):

        # first sequence
        out = self.conv1(x)
        out = F.relu(out)
        out = self.maxPool1(out)
        out = self.batch1(out)

        # second sequence
        out = self.conv2(out)
        out = F.relu(out)
        out = self.maxPool2(out)
        out = self.batch2(out)

        # Linear

        out = out.view(out.size(0), 5 * 5 * 64)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        out = F.relu(out)
        out = self.fc5(out)

        return out
