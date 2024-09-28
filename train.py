from models import MNIST_CNN

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm

# training dataset
train_ds = datasets.MNIST(
    "./data",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]),
)

epochs = 15
batch_size = 100
validation_split = 0.1
shuffle_dataset = True

dataset_size = len(train_ds)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
random_seed = 2

if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)


train_indices, val_indices = indices[split:], indices[:split]


train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=batch_size, sampler=train_sampler
)
validation_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=batch_size, sampler=valid_sampler
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    ),
    batch_size=batch_size,
    shuffle=True,
)


if torch.cuda.is_available():
    model = MNIST_CNN().cuda()
    print("Loading to GPU")
    training_on = "GPU"
else:  # loading to CPU
    model = MNIST_CNN().cpu()
    print("Loading to CPU")
    training_on = "CPU"


optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

train_errors = list()
train_acc = list()
val_errors = list()
val_acc = list()
n_train = len(train_loader) * batch_size
n_val = len(validation_loader) * batch_size


print("Starting Training Process...")
print("Dataset Sizes:")
print(f"Training Set: {len(train_indices)} samples")
print(f"Validation Set: {len(val_indices)} samples")
print(f"Testing Set: {len(test_loader.dataset)} samples")
print("\nTraining Parameters:")
print(f"Epochs: {epochs}")
print(f"Batch Size: {batch_size}")
print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")
print(f"Optimizer: {optimizer.__class__.__name__}")
print(f"Loss Function: {criterion.__class__.__name__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Training on : {training_on}")

print()

for i in range(epochs):
    total_loss = 0
    total_acc = 0
    c = 0

    # Training phase
    model.train()  # Set the model to training mode
    with tqdm(
        total=len(train_loader), desc=f"Training Epoch {i + 1}", unit="batch"
    ) as pbar:
        for images, labels in train_loader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            else:
                images = images.cpu()
                labels = labels.cpu()

            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += torch.sum(torch.max(output, dim=1)[1] == labels).item() * 1.0
            c += 1

            pbar.set_postfix(
                loss=total_loss / (c + 1), accuracy=total_acc / (c + 1)
            )  # Update progress bar
            pbar.update(1)  # Increment progress bar

    # Validation phase
    total_loss_val = 0
    total_acc_val = 0
    c = 0
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation for validation
        for images, labels in validation_loader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            else:
                images = images.cpu()
                labels = labels.cpu()

            output = model(images)
            loss = criterion(output, labels)

            total_loss_val += loss.item()
            total_acc_val += (
                torch.sum(torch.max(output, dim=1)[1] == labels).item() * 1.0
            )
            c += 1

    # Compute average losses and accuracies
    avg_loss_train = total_loss / len(train_loader)
    avg_acc_train = total_acc / len(train_loader.dataset)
    avg_loss_val = total_loss_val / len(validation_loader)
    avg_acc_val = total_acc_val / len(validation_loader.dataset)

    # Print training and validation results
    print(
        f"Epoch [{i + 1}/{epochs}], Train Loss: {avg_loss_train:.4f}, Train Acc: {avg_acc_train:.4f}, "
        f"Val Loss: {avg_loss_val:.4f}, Val Acc: {avg_acc_val:.4f}",
        "\n",
    )


model_save_path = 'models/mnist_model_v1.pth'  # Specify the path where you want to save the model
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')
