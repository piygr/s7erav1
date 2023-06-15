import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms

'''
>>>>>>>>>>>>>>>
target - 
1. Building data loaders, test & train data sets and train & test loop
2. Also, setting basic skeleton with working model (without shape, size errors in model) 
3. Working model should be able to reach 98-99% accuracy on the dataset with the skeleton model

Result -
1. 390K+ parameters
2. Best training accuracy - 99.94%
3. Best test accuracy - 99.41%

Analysis -
1. Too many parameters, need lighter model
2. Overfitting
<<<<<<<<<<<<<<<<


>>>>>>>>>>>>>>>
target -
1. Building a lighter model with params under 30k

Result -
1. ~26K parameters
2. Best training accuracy - 99.84
3. Best test accuracy - 99.23%

Analysis -
1. Skeleton working, need to further reduce the params
2. Overfitting
<<<<<<<<<<<<<<<<
'''


class Model1(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Model1, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3),    # rf 3  -> 26x26
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 3),  # rf 5   -> 24x24
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool2d(2, 2) #rf 6 -> 12x12

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3),    #rf 10 -> 10x10
            nn.ReLU()
        )

        self.pool2 = nn.MaxPool2d(2, 2) #rf 12 -> 5x5

        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3),       #rf 20 -> 3x3
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3),  # rf 28 -> 1x1
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(32, 10, 1)
        )


    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), 10)

        return F.log_softmax(x, dim=1)


train_transforms = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])

# Test data transformations
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

def get_scheduler(optimizer):
    return optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1, verbose=True)