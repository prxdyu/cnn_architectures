## RestNet Implementation using PyTorch
This repository contains an implementation of the ResNet50 architecture using PyTorch. ResNet50 is a deep convolutional neural network that is widely used for various computer vision tasks such as image classification, object detection, and image segmentation.

## Introduction
ResNet50 is a variant of the ResNet architecture introduced by Kaiming He et al. in their paper "Deep Residual Learning for Image Recognition". It consists of 50 layers and is known for its ability to train very deep neural networks effectively. This implementation provides a modular and flexible way to build ResNet50 networks for various computer vision tasks.

## Features:
-   **Flexible Usage**: Users can create ResNet50 models with different numbers of input channels and classes, making it versatile for different tasks and datasets.

## Usage:
To use the ResNet50 implementation in your project, follow these steps:
- Install the required dependencies, including PyTorch.
- Import the ResNet50 function from the provided code.
- Instantiate the ResNet50 model with appropriate parameters,
    - **img_channels**: number of input channels
        - 1 if the input images are in grayscale
        - 3 if input images are colored)
    - **num_classes**: the number of classes in your classification task.
- Train the model on your dataset and evaluate its performance.
### Example
      import torch
      from resnet import ResNet50
      
      # Define the number of input channels and classes
      num_channels = 3
      num_classes = 1000
      
      # Instantiate the ResNet50 model
      model = ResNet50(num_channels, num_classes)
      
      # Load your dataset and train the model
      # Add your training code here...
      
      # Evaluate the model
      # Add your evaluation code here...
