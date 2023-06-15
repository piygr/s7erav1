import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms

'''
>>>>>>>>>>>>>>>
target -
1. Building a lighter model with params under 8k

Result -
1. 6.7k parameters
2. Best training accuracy - 99.45 (20th epoch)
3. Best test accuracy - 98.97% (13th epoch)

Analysis -
1. Good model and can be pushed further
2. Overfitting
<<<<<<<<<<<<<<<<

>>>>>>>>>>>>>>>
target - 
1. Add normalisation, BatchNorm to push model efficiency

Result -
1. ~6.9k params
2. Best training accuracy - 99.74%
3. Best test accuracy - 99.21%

Analysis -
1. Still there's overfitting
2. Model efficiency can't be pushed further
<<<<<<<<<<<<<<<<

>>>>>>>>>>>>>>>
target -
1. Add regularization (Dropout) to get rid of overfitting

Result -
1. Best training accuracy - 98.90 (19th epoch)
2. Best test accuracy - 99.30% (18th epoch)

Analysis -
1. Underfitting but that's because of regularisation, Good
2. Model can't be pushed further with current capacity
'''


class Model2(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Model2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),    # rf 3  -> 28x28
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Dropout2d(0.05)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1),   #rf 5   -> 28x28
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout2d(0.05)
        )

        self.pool1 = nn.MaxPool2d(2, 2) #rf 6 -> 12x12

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 12, 3),    #rf 10 -> 12x12
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout2d(0.05)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(12, 12, 3),  # rf 14 -> 10x10
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout2d(0.05),
        )

        self.pool2 = nn.MaxPool2d(2, 2) #rf 16 -> 5x5

        self.conv5 = nn.Sequential(
            nn.Conv2d(12, 16, 3),       #rf 24 -> 3x3
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.05)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(16, 16, 3),      #rf 28 -> 1x1
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.05)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(16, 10, 1)
        )


    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
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