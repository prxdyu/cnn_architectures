# LeNet Implementation using PyTorch

This repository contains an implementation of the LeNet architecture using PyTorch, a popular deep learning framework. LeNet, proposed by Yann LeCun et al. in 1998, is one of the pioneering convolutional neural network (CNN) architectures and played a significant role in the advancement of deep learning.

## Introduction

LeNet was introduced by Yann LeCun et al. in their paper "Gradient-Based Learning Applied to Document Recognition". It consists of a simple yet effective architecture designed for handwritten digit classification. This implementation provides a modular and flexible way to build LeNet networks for various image classification tasks.

## Key Features

- **LeNet Architecture**: The repository includes the implementation of the LeNet architecture, which comprises convolutional layers with ReLU activation functions, max-pooling layers, and fully connected layers.
- **MNIST Dataset**: The model is trained and evaluated using the MNIST dataset, a widely used benchmark dataset in the field of computer vision and deep learning.
- **Training Script**: A training script is provided to train the LeNet model on the MNIST dataset. It includes data loading, model training, and evaluation steps.
- **Optimizer Selection**: The implementation allows users to choose between different optimizers for training the model. By default, it uses the SGD optimizer, but it can be easily switched to other optimizers such as Adam or RMSProp.

## Usage

To use the LeNet implementation in your project, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies specified in the `requirements.txt` file.
3. Run the training script to train the LeNet model on the MNIST dataset.
4. Evaluate the trained model's performance on the test set and analyze the results.

### Example

```python
# Importing the libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
import os

# Implementing LeNet architecture
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = F.relu
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=1, padding=0)
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def evaluate(self, loader):
        num_samples = 0
        num_correct = 0
        self.eval()
        with torch.no_grad():
            for x, y_true in loader:
                x = x.to(device='cpu')
                y_true = y_true.to(device='cpu')
                logits = self(x)
                _, y_pred = logits.max(dim=1)
                num_correct += (y_pred == y_true).sum()
                num_samples += y_pred.size(0)
            accuracy = float(num_correct) / float(num_samples)
        return accuracy

# Example of using the LeNet model

# Define a transform to preprocess the images
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# Load your dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Instantiate the LeNet model
model = LeNet()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):  # number of epochs
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'lenet.pth')

# To evaluate the model, load it and run inference on your test dataset
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluate the model
accuracy = model.evaluate(test_loader)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
