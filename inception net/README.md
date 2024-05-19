# InceptionNet Implementation with PyTorch

This repository contains an implementation of the InceptionNet architecture using PyTorch. InceptionNet, also known as GoogLeNet, is a deep convolutional neural network introduced by Google in 2014. It is famous for its inception modules, which allow for efficient and parallel processing of information at different scales.

### Features:
- Implementation of the InceptionNet architecture in PyTorch.
- Modular design with customizable hyperparameters.
- Designed for image classification tasks with 1000 classes.
- Includes the main model definition (InceptionNet.py) and a sample jupyter notebook for model evaluation (InceptionNet Evaluation.ipynb)
- InceptionNet(input_channels,num_classes)
  - **input_channels**= No of Channels in the images in your dataset "1" if grayscale "3" if RGB
  - **num_classes**= No of Classes in your classification task  

### Usage:
- Clone the repository to your local machine:
  
        git clone https://github.com/your-username/your-repository.git
- Install the required dependencies, including PyTorch:

       pip install torch torch
- Import the InceptionNet model in your Python script:

      from InceptionNet import InceptionNet
  
      # creating an instance of the model
      model=InceptionNet()
      .... 
- Explore the provided implementation and modify it as needed for your specific tasks.
