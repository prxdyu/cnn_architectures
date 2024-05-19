import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader



class VGG16(nn.Module):

  """
  Takes input image of size 224x224
     Parameters:

        in_channels   : no of input channels in the image 1 if input image is in grayscal 3 if input image is in color (defaults to 3)
        num_classes   : no of classes in your classification setup (defaults to 1000)
  """

  # defining the architecture as list
  architecture=[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M']

  def __init__(self,in_channels=3,num_classes=1000):

    super(VGG16,self).__init__()
    self.in_channels=in_channels
    self.num_classes=num_classes
    # creating convolutional layers
    self.conv_layers=self.create_conv_layers(self.architecture)
    # creating FC layers
    self.fc_layers=nn.Sequential(nn.Linear(in_features=7*7*512,out_features=4096),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(in_features=4096,out_features=4096),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(in_features=4096,out_features=self.num_classes))


  def create_conv_layers(self,architecture):

    # defining empty list to store the layers of vgg16
    layers=[]
    in_channels=self.in_channels
    # iterating through the architecture list to create layers
    for i in architecture:

      # if the i is an int then it is a conv layer having i no of kernels
      if type(i)==int:
        out_channels=i
        # creating Conv layer with batchnorm and relu and appending it to the layers list
        layers+=[nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
                      nn.BatchNorm2d(num_features=i),
                      nn.ReLU()]
        # updating the in_channel for the next layer (out_channels in the current layer ====> in_channel for next layer)
        in_channels=i

      # if i is M then it is a Max Pool layer
      elif i=="M":
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))

    # creating the model using the sequential API and returning the model
    return nn.Sequential(*layers)


  def forward(self,x):
    # passing images through the convolutional layers
    x=self.conv_layers(x)
    # passing images through the FC layers
    x = x.view(x.size(0), -1)
    x=self.fc_layers(x)
    return x


 