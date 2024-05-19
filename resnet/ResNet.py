

# importing the required libraries
import torch
import torch.nn as nn


class block(nn.Module):
  """
  Defining a class for the block
  A block consists of
   - 1x1 Convolutions
   - 3x3 Convolutions
   - 1x1 Convolutions
  """
  def __init__(self,in_channels,out_channels,identity_downsample=None,stride=1):
    super(block,self).__init__()

    self.expansion=4
    self.relu=nn.ReLU()
    self.identity_downsample=identity_downsample

    # 1x1 convolutions
    self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0)
    self.bn1=nn.BatchNorm2d(out_channels)

    # 3x3 convoltuions (here the out_channels is same as the in_channels) refer diagram
    self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1)
    self.bn2=nn.BatchNorm2d(out_channels)

    # 1x1 convolution (here out_channels=4*in_channels)
    self.conv3=nn.Conv2d(out_channels,self.expansion*out_channels,kernel_size=1,stride=1,padding=0)
    self.bn3=nn.BatchNorm2d(self.expansion*out_channels)


  def forward(self,x):

    # making a copy of input x
    identity=x

    # passing the input through the block
    x=self.conv1(x)  # 1x1 conv
    x=self.bn1(x)
    x=self.relu(x)

    x=self.conv2(x) # 3x3 conv
    x=self.bn2(x)
    x=self.relu(x)

    x=self.conv3(x) # 1x1 conv
    x=self.bn3(x)
    x=self.relu(x)

    # checking if the input for the block to be modified to match the dimension of output for making residual connections
    if self.identity_downsample is not None:
      identity=self.identity_downsample(identity) 

    # making residual connection
    x+=identity
    x=self.relu(x)
    return x

class ResNet(nn.Module):

  def __init__(self,block,img_channels,num_classes):
    super(ResNet,self).__init__()

    # setting the in_channels=64 as the no of channels input to our first block will be 64
    self.in_channels=64
    #reps: list of how many times we have to resue the each block in the resnet
    reps=[3,4,6,3]   # we are using the first block 3 times so first element of the list is 3

    # 64 7x7 conv
    self.conv1=nn.Conv2d(img_channels,64,kernel_size=7,stride=2,padding=3)
    self.bn1=nn.BatchNorm2d(64)
    self.relu=nn.ReLU()
    self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

    """ layer1 consists of 3X block1
        layer2 consists of 4X block2
        layer3 consists of 6X block3
        layer4 consists of 3X block4 """

    # creating the layers
    self.layer1=self.make_resnet_layer(block,reps[0],out_channels=64,stride=1) # only the first layer has stride=1 since the blocks in this layer reatains the input height and width
    self.layer2=self.make_resnet_layer(block,reps[1],out_channels=128,stride=2)
    self.layer3=self.make_resnet_layer(block,reps[2],out_channels=256,stride=2)
    self.layer4=self.make_resnet_layer(block,reps[3],out_channels=512,stride=2)
    # creating the FC layers
    self.avgpool=nn.AdaptiveAvgPool2d((1,1))
    self.fc1=nn.Linear(512*4,num_classes)



  # function which creates resnet layers
  def make_resnet_layer(self,block,repetition,out_channels,stride):
    """ block:           function which creates block
        repetition:      no of times to repeat the block
        out_channels:    output_channels/4 from the block """

    identity_downsample=None
    layers=[]

    # checking condition (if the dimensions or channels of the input tensor is changed by the block then  create the identity block which makes the shape of input tensor compatiable with output tensor of the block to make residual connection)
    if  (stride!=1) or (self.in_channels!=out_channels*4):
      identity_downsample=nn.Sequential(nn.Conv2d(self.in_channels,out_channels*4,kernel_size=1,stride=stride),
                                        nn.BatchNorm2d(out_channels*4)
                                        )
    # creating the first repetition of block of the resnet layer and appending it to the layers list
    layers.append(block(self.in_channels,out_channels,identity_downsample,stride))
    # the out_channels of this repetition of the block will be the in_channels for the subsequent repetitions
    self.in_channels=out_channels*4

    # appending the rest of the repetitions of the block
    for i in range(repetition-1):
      layers.append(block(self.in_channels,out_channels)) # all of the repetitions except first don't need identity block
    return nn.Sequential(*layers)


  def forward(self,x):

    x=self.conv1(x)
    x=self.bn1(x)
    x=self.relu(x)
    x=self.maxpool(x)
    x=self.layer1(x)
    x=self.layer2(x)
    x=self.layer3(x)
    x=self.layer4(x)

    x=self.avgpool(x)
    x=x.reshape(x.shape[0],-1)
    x=self.fc1(x)

    return x

# creating a helper function which creates resnet50 model
def ResNet50(img_channels, num_classes):
    return ResNet(block, img_channels, num_classes)




