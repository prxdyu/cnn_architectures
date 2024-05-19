

# importng the libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
import os

# implementing LeNet architecture
class LeNet(nn.Module):
  def __init__(self):
    super(LeNet,self).__init__()  # make our custom class LeNet to inherit __init__ of nn.module
    self.relu=F.relu
    self.pool=nn.AvgPool2d(kernel_size=(2,2))
    self.conv1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=(5,5),stride=1,padding=0) #padding:0 ==> no padding , padding:1 ==> 'same'
    self.conv2=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=(5,5),stride=1,padding=0)
    self.conv3=nn.Conv2d(in_channels=16,out_channels=120,kernel_size=(5,5),stride=1,padding=0)
    self.linear1=nn.Linear(120,84)
    self.linear2=nn.Linear(84,10)


  def forward(self,x):
    x=self.relu(self.conv1(x))
    x=self.pool(x)
    x=self.relu(self.conv2(x))
    x=self.pool(x)
    x=self.relu(self.conv3(x))
    # reshaping  batchsize X 120 X 1 X 1 ====> batchsize X 120
    x=x.reshape(x.shape[0],-1)
    x=self.relu(self.linear1(x))
    x=self.linear2(x)
    return x

  def evaluate(self,loader):
    num_samples=0
    num_correct=0
    # setting model to evaluation mode  #Batch normalization layers behave differently during training and evaluation. During training, they use batch statistics for normalization, but during evaluation, they use population statistics.Dropout layers also behave differently during training and evaluation. During training, they randomly drop units, but during evaluation, they keep all units and adjust the weights accordingly.model.eval() sets the model to evaluation mode, which ensures that batch normalization and dropout layers behave in the appropriate way during inference.
    self.eval()
    # we don't want to compute gradients while evaluating
    with torch.no_grad():
      for x,y_true in loader:
        # moving x and y to the CPU
        x=x.to(device='cpu')
        y_true=y_true.to(device='cpu')
        logits=self(x)
        # getting labels
        _,y_pred=logits.max(dim=1)
        num_correct+=(y_pred==y_true).sum()
        num_samples+=y_pred.size(0)
        # computing the accurcy
      accuracy=float(num_correct)/float(num_samples)
      # assert(accuracy==float)
    # print(f"The accuracy is {accuracy}")
    return accuracy


