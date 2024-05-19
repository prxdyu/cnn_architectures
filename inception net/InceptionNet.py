

import torch
import torch.nn as nn

# creating a convolution block which creates ConvLayers
class conv_block(nn.Module):

  def __init__(self,in_channels,out_channels,**kwargs):
    super(conv_block,self).__init__()
    self.conv=nn.Conv2d(in_channels,out_channels,**kwargs)
    self.relu=nn.ReLU()
    self.batchnorm=nn.BatchNorm2d(out_channels)

  def forward(self,x):
    x=self.batchnorm(self.relu(self.conv(x)))
    return x

# This class represents an Inception Block
class InceptionBlock(nn.Module):

  def __init__(self,in_channels,out_1x1,reduce_3x3,out_3x3,reduce_5x5,out_5x5,out_1x1_branch4):
    super(InceptionBlock,self).__init__()

    # branch1
    self.branch1=conv_block(in_channels,out_1x1,kernel_size=1)
    # branch2  (1x1 => 3x3)
    self.branch2=nn.Sequential(conv_block(in_channels,reduce_3x3,kernel_size=1),
                               conv_block(reduce_3x3,out_3x3,kernel_size=3,padding=1))
    # branch3 (1x1 => 5x5 )
    self.branch3=nn.Sequential(conv_block(in_channels,reduce_5x5,kernel_size=1),
                               conv_block(reduce_5x5,out_5x5,kernel_size=5,padding=2))
    # branch4 (3x3pool => 1x1)
    self.branch4=nn.Sequential(nn.MaxPool2d(kernel_size=3,padding=1,stride=1),
                               conv_block(in_channels,out_1x1_branch4,kernel_size=1))

  def forward(self,x):
    # forward prop through 4 branches and concatenating the outputs along filter dimension
    # batches X (filters/channels) X 28 X 28
    concatenated=torch.cat([self.branch1(x),self.branch2(x),self.branch3(x),self.branch4(x)],dim=1)
    return concatenated

# This class represents out whole InceptionNet Architecture
class InceptionNet(nn.Module):

  def __init__(self,in_channels=3,num_classes=1000):
    super(InceptionNet,self).__init__()

    self.in_channels=in_channels
    self.num_classes=num_classes

    self.conv1=conv_block(in_channels=in_channels,out_channels=64,kernel_size=7,stride=2,padding=3)
    self.maxpool1=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
    self.conv2=conv_block(in_channels=64,out_channels=192,kernel_size=3,stride=1,padding=1)
    self.maxpool2=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

    # inception_block 3a     # refer table in the paper for the below arguments
    self.inception_3a=InceptionBlock(in_channels=192,out_1x1=64,reduce_3x3=96,out_3x3=128,reduce_5x5=16,out_5x5=32,out_1x1_branch4=32)
    # inception block 3b
    self.inception_3b=InceptionBlock(256,128,128,192,32,96,64)

    self.maxpool3=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

    # inception block 4a
    self.inception_4a=InceptionBlock(480,192,96,208,16,48,64)
    # inception block 4b
    self.inception_4b=InceptionBlock(512,160,128,192,32,96,64)
    # inception block 4c
    self.inception_4c=InceptionBlock(512,128,128,256,24,64,64)
    # inception block 4d
    self.inception_4d=InceptionBlock(512,112,144,288,32,64,64)
    # inception block 4e
    self.inception_4e=InceptionBlock(528,256,160,320,32,128,128)

    self.maxpool4=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

    # inception block 5a
    self.inception_5a=InceptionBlock(832,256,160,320,32,128,128)
    # inception block 5b
    self.inception_5b=InceptionBlock(832,384,192,384,48,128,128)

    self.avgpool=nn.AvgPool2d(kernel_size=7,stride=1)
    self.dropout=nn.Dropout(p=0.4)
    self.fc1=nn.Linear(1024,1000)


  def forward(self,x):

    x=self.conv1(x)
    x=self.maxpool1(x)
    x=self.conv2(x)
    x=self.maxpool2(x)

    x=self.inception_3a(x)
    x=self.inception_3b(x)
    x=self.maxpool3(x)

    x=self.inception_4a(x)
    x=self.inception_4b(x)
    x=self.inception_4c(x)
    x=self.inception_4d(x)
    x=self.inception_4e(x)
    x=self.maxpool4(x)

    x=self.inception_5a(x)
    x=self.inception_5b(x)
    x=self.avgpool(x)
    x=self.dropout(x)
    x = torch.flatten(x, 1)
    x=self.fc1(x)

    return x
