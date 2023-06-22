import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms

'''
>>>>>>>>>>>>>>>
target -
1. Use GAP

Result -
1. 4.5k parameters
2. Best training accuracy - 98.74 (20th epoch)
3. Best test accuracy - 99.28% (18th Epoch)

Analysis -
1. Need to add capacity to reach 99.4 goal
2. No Overfitting
<<<<<<<<<<<<<<<<

>>>>>>>>>>>>>>>
target - 
1. Add FC post GAP layer and see

Result -
1. Model parameters - 6.5k
2. Best training accuracy - 99.11% (20th epoch)
3. Best test accuracy - 99.36% (20th epoch)

Analysis -
1. Underfitting but it's fine. 
2. Need to playaround with image transforms to make training difficult
<<<<<<<<<<<<<<<<

>>>>>>>>>>>>>>>
target -
1. Add transformations to input dataset
2. Need to add rotation wit -15 - 15 degree

Result -
1. Best training accuracy - 98.36 (17th epoch)
2. Best test accuracy - 99.34% (12th epoch)

Analysis -
1. Training is difficult enough
2. Need to reduce epochs
<<<<<<<<<<<<<<<<

>>>>>>>>>>>>>>>
target - 
1. Achieve 99.4% accuracy within 15 epochs. Trying out ReduceLROnPlateau
2. Learning rate 0.1, patience=2, threshold = 0.001

Result -
1. Best training accurancy - 98.73 (14th epoch)
2. Best test accuracy - 99.41 (12th epoch)

Analysis -
1. Learning rate & batch size directly affects number of epochs 

<<<<<<<<<<<<<<<<
'''


class Model3(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Model3, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),  # rf 3  -> 28x28
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout2d(0.05)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1),  # rf 5   -> 28x28
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout2d(0.05)
        )

        self.pool1 = nn.MaxPool2d(2, 2)  # rf 6 -> 14x14

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 12, 3),  # rf 10 -> 12x12
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout2d(0.05)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(12, 16, 3),  # rf 14 -> 10x10
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.05),
        )

        self.pool2 = nn.MaxPool2d(2, 2)  # rf 16 -> 5x5

        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 20, 3),  # rf 24 -> 3x3
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout2d(0.05)
        )

        '''
        self.conv6 = nn.Sequential(
            nn.Conv2d(16, 16, 3),  # rf 28 -> 1x1
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.05)
        )
        

        self.conv7 = nn.Sequential(
            nn.Conv2d(16, 10, 1)            # 10x1x1
        )'''

        self.gap = nn.Sequential(
            nn.AvgPool2d(3)                 # 24 + (3-1)*4
        )

        self.fc = nn.Linear(20, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        #x = self.conv6(x)
        #x = self.conv7(x)
        x = self.gap(x)
        #print(x.size())
        x = x.view(x.size(0), 20)
        #print(x.size())
        x = self.fc(x)

        return F.log_softmax(x, dim=1)


train_transforms = transforms.Compose([
    transforms.RandomRotation((-10., 10.), fill=0),
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
    #return optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1, verbose=True)
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1, threshold=0.001, threshold_mode='abs', eps=0.001, verbose=True)