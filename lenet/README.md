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

# importing custom made LeNet
from LeNet import LeNet


# Instantiate the LeNet model
model = LeNet()
......
