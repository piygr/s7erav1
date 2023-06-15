# Session 7 Assignment
Model to detect handwritten digits, trained on MNIST dataset of 60,000 images.

**Goal is to create a model with**
- 99.4% validation accuracy with consistency
- Less than 8k Parameters
- Less than 15 Epochs
(Optional): a Fully connected layer, have used GAP.

## utils.py
The file contains utility & helper functions needed for training & for evaluating our model.

## S7.ipynb
The file is an IPython notebook. The notebook imports helper functions from utils.py and Model class from Model_1.py, Model_2.py & Model_3.py.

## Model_1.py
<table>
        <tr>
                <th>Target</th>
                <th>Result</th>
                <th>Analysis</th>
        </tr>
        <tr>
                <td>
                        <ol>
                        <li>Building data loaders, test & train data sets and train & test loop</li>
                        <li>Also, setting basic skeleton with working model (without shape, size errors in model) </li>
                        <li>Working model should be able to reach 98-99% accuracy on the dataset with the skeleton model</li>
                        </ol>
                </td>
                <td>
                        <ol>
                        <li>390K+ parameters </li>
                        <li>Best training accuracy - 99.94% </li>
                        <li>Best test accuracy - 99.41% </li>
                        </ol>
                </td>
                <td>
                        <ol>
                        <li>Too many parameters, need lighter model </li>
                        <li>Overfitting </li>
                        </ol>
                </td>
        </tr>
        <tr>
                <td>
                        <ol>
                                <li>Building a lighter model with params under 30k</li>
                        </ol>
                </td>
                <td>
                        <ol>
                                <li>~26K parameters</li>
                                <li>Best training accuracy - 99.84</li>
                                <li>Best test accuracy - 99.23%</li>
                        </ol>
                </td>
                <td>
                        <ol>
                                <li>Skeleton working, need to further reduce the params</li>
                                <li>Overfitting</li>
                        </ol>
                </td>
        </tr>
</table>

## Model_2.py
<table>
        <tr>
                <th>Target</th>
                <th>Result</th>
                <th>Analysis</th>
        </tr>
        <tr>
                <td>
                        <ol>
                        <li>Building a lighter model with params under 8k</li>
                        </ol>
                </td>
                <td>
                        <ol>
                        <li>6.7k parameters</li>
                        <li>Best training accuracy - 99.45 (20th epoch)</li>
                        <li>Best test accuracy - 98.97% (13th epoch)</li>
                        </ol>
                </td>
                <td>
                        <ol>
                        <li>Good model and can be pushed further</li>
                        <li>Overfitting </li>
                        </ol>
                </td>
        </tr>
        <tr>
                <td>
                        <ol>
                                <li>Add normalisation, BatchNorm to push model efficiency</li>
                        </ol>
                </td>
                <td>
                        <ol>
                                <li>~6.9k params</li>
                                <li>Best training accuracy - 99.74%</li>
                                <li>Best test accuracy - 99.21%</li>
                        </ol>
                </td>
                <td>
                        <ol>
                                <li>Still there's overfitting</li>
                                <li>Model efficiency can't be pushed further</li>
                        </ol>
                </td>
        </tr>
         <tr>
                <td>
                        <ol>
                                <li>Add regularization (Dropout) to get rid of overfitting</li>
                        </ol>
                </td>
                <td>
                        <ol>
                                <li>Best training accuracy - 98.90 (19th epoch)/li>
                                <li>Best test accuracy - 99.30% (18th epoch)</li>
                        </ol>
                </td>
                <td>
                        <ol>
                                <li>Underfitting but that's because of regularisation, Good</li>
                                <li>Model can't be pushed further with current capacity</li>
                        </ol>
                </td>
        </tr>
</table>

## Model_3.py
<table>
        <tr>
                <th>Target</th>
                <th>Result</th>
                <th>Analysis</th>
        </tr>
        <tr>
                <td>
                        Add GAP & remove last layer
                </td>
                <td>
                        <ol>
                        <li>4.5k parameters</li>
                        <li>Best training accuracy - 98.74 (20th epoch)</li>
                        <li>Best test accuracy - 99.28% (18th Epoch)</li>
                        </ol>
                </td>
                <td>
                        <ol>
                        <li>Need to add capacity to reach 99.4 goal</li>
                        <li>No overfitting </li>
                        </ol>
                </td>
        </tr>
        <tr>
                <td>
                        Add FC post GAP layer and see
                </td>
                <td>
                        <ol>
                                <li>Model parameters - 6.5k</li>
                                <li>Best training accuracy - 99.11% (20th epoch)</li>
                                <li>Best test accuracy - 99.36% (20th epoch)</li>
                        </ol>
                </td>
                <td>
                        <ol>
                                <li>Underfitting but it's fine. </li>
                                <li>Need to playaround with image transforms to make training difficult</li>
                        </ol>
                </td>
        </tr>
         <tr>
                <td>
                        <ol>
                                <li>Add transformations to input dataset</li>
                                <li>Need to add rotation between (-10) to (10) degree</li>
                        </ol>
                </td>
                <td>
                        <ol>
                                <li>Best training accuracy - 98.36 (17th epoch)/li>
                                <li>Best test accuracy - 99.34% (12th epoch)</li>
                        </ol>
                </td>
                <td>
                        <ol>
                                <li>Training is difficult enough</li>
                                <li>Need to reduce epochs using right LR strategy</li>
                        </ol>
                </td>
        </tr>
        <tr>
                <td>
                        <ol>
                                <li>Achieve 99.4% accuracy within 15 epochs. Trying out ReduceLROnPlateau (Learning rate 0.1, patience=2, threshold = 0.001)</li>
                        </ol>
                </td>
                <td>
                        <ol>
                                <li>Best training accurancy - 98.73 (14th epoch)/li>
                                <li>Best test accuracy - 99.41 (12th epoch)</li>
                        </ol>
                </td>
                <td>
                        Learning rate & batch size directly affects number of epochs
                </td>
        </tr>
</table>

Below is the model summary -
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              80
              ReLU-2            [-1, 8, 28, 28]               0
       BatchNorm2d-3            [-1, 8, 28, 28]              16
         Dropout2d-4            [-1, 8, 28, 28]               0
            Conv2d-5            [-1, 8, 28, 28]             584
              ReLU-6            [-1, 8, 28, 28]               0
       BatchNorm2d-7            [-1, 8, 28, 28]              16
         Dropout2d-8            [-1, 8, 28, 28]               0
         MaxPool2d-9            [-1, 8, 14, 14]               0
           Conv2d-10           [-1, 16, 14, 14]           1,168
             ReLU-11           [-1, 16, 14, 14]               0
      BatchNorm2d-12           [-1, 16, 14, 14]              32
        Dropout2d-13           [-1, 16, 14, 14]               0
           Conv2d-14           [-1, 16, 14, 14]           2,320
             ReLU-15           [-1, 16, 14, 14]               0
      BatchNorm2d-16           [-1, 16, 14, 14]              32
        Dropout2d-17           [-1, 16, 14, 14]               0
           Conv2d-18           [-1, 32, 12, 12]           4,640
             ReLU-19           [-1, 32, 12, 12]               0
      BatchNorm2d-20           [-1, 32, 12, 12]              64
        Dropout2d-21           [-1, 32, 12, 12]               0
        MaxPool2d-22             [-1, 32, 6, 6]               0
           Conv2d-23             [-1, 32, 4, 4]           9,248
             ReLU-24             [-1, 32, 4, 4]               0
      BatchNorm2d-25             [-1, 32, 4, 4]              64
        Dropout2d-26             [-1, 32, 4, 4]               0
           Conv2d-27             [-1, 10, 4, 4]             330
================================================================
Total params: 18,594
Trainable params: 18,594
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.75
Params size (MB): 0.07
Estimated Total Size (MB): 0.83
----------------------------------------------------------------
```

We can monitor our model performance while it's getting trained. The output looks like this - 
```
Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 1
Train: Loss=0.1356 Batch_id=937 Accuracy=92.10: 100%|██████████| 938/938 [00:47<00:00, 19.96it/s]
Test set: Average loss: 0.0456, Accuracy: 9868/10000 (98.68%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 2
Train: Loss=0.0651 Batch_id=937 Accuracy=97.64: 100%|██████████| 938/938 [00:49<00:00, 19.09it/s]
Test set: Average loss: 0.0368, Accuracy: 9881/10000 (98.81%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 3
Train: Loss=0.0078 Batch_id=937 Accuracy=98.07: 100%|██████████| 938/938 [00:40<00:00, 23.30it/s]
Test set: Average loss: 0.0215, Accuracy: 9930/10000 (99.30%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 4
Train: Loss=0.0287 Batch_id=937 Accuracy=98.25: 100%|██████████| 938/938 [00:44<00:00, 21.14it/s]
Test set: Average loss: 0.0206, Accuracy: 9934/10000 (99.34%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 5
Train: Loss=0.0379 Batch_id=937 Accuracy=98.53: 100%|██████████| 938/938 [00:44<00:00, 21.15it/s]
Test set: Average loss: 0.0208, Accuracy: 9930/10000 (99.30%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 6
Train: Loss=0.0112 Batch_id=937 Accuracy=98.54: 100%|██████████| 938/938 [00:45<00:00, 20.62it/s]
Test set: Average loss: 0.0185, Accuracy: 9937/10000 (99.37%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 7
Train: Loss=0.0785 Batch_id=937 Accuracy=98.61: 100%|██████████| 938/938 [00:45<00:00, 20.56it/s]
Test set: Average loss: 0.0190, Accuracy: 9942/10000 (99.42%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 8
Train: Loss=0.0279 Batch_id=937 Accuracy=98.87: 100%|██████████| 938/938 [00:43<00:00, 21.38it/s]
Test set: Average loss: 0.0150, Accuracy: 9948/10000 (99.48%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 9
Train: Loss=0.1910 Batch_id=937 Accuracy=98.95: 100%|██████████| 938/938 [00:43<00:00, 21.52it/s]
Test set: Average loss: 0.0149, Accuracy: 9954/10000 (99.54%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 10
Train: Loss=0.0464 Batch_id=937 Accuracy=98.92: 100%|██████████| 938/938 [00:43<00:00, 21.32it/s]
Test set: Average loss: 0.0145, Accuracy: 9950/10000 (99.50%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 11
Train: Loss=0.0013 Batch_id=937 Accuracy=99.00: 100%|██████████| 938/938 [00:47<00:00, 19.81it/s]
Test set: Average loss: 0.0143, Accuracy: 9952/10000 (99.52%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 12
Train: Loss=0.1712 Batch_id=937 Accuracy=99.06: 100%|██████████| 938/938 [00:44<00:00, 21.30it/s]
Test set: Average loss: 0.0138, Accuracy: 9953/10000 (99.53%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 13
Train: Loss=0.0084 Batch_id=937 Accuracy=99.06: 100%|██████████| 938/938 [00:44<00:00, 21.24it/s]
Test set: Average loss: 0.0135, Accuracy: 9955/10000 (99.55%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 14
Train: Loss=0.0052 Batch_id=937 Accuracy=99.13: 100%|██████████| 938/938 [00:44<00:00, 21.00it/s]
Test set: Average loss: 0.0132, Accuracy: 9956/10000 (99.56%)

Adjusting learning rate of group 0 to 1.0000e-04.
Epoch 15
Train: Loss=0.0142 Batch_id=937 Accuracy=99.06: 100%|██████████| 938/938 [00:43<00:00, 21.55it/s]
Test set: Average loss: 0.0131, Accuracy: 9955/10000 (99.55%)

Adjusting learning rate of group 0 to 1.0000e-04.
Epoch 16
Train: Loss=0.0187 Batch_id=937 Accuracy=98.97: 100%|██████████| 938/938 [00:44<00:00, 20.92it/s]
Test set: Average loss: 0.0132, Accuracy: 9957/10000 (99.57%)

Adjusting learning rate of group 0 to 1.0000e-04.
Epoch 17
Train: Loss=0.0015 Batch_id=937 Accuracy=99.11: 100%|██████████| 938/938 [00:45<00:00, 20.62it/s]
Test set: Average loss: 0.0129, Accuracy: 9960/10000 (99.60%)

Adjusting learning rate of group 0 to 1.0000e-04.
Epoch 18
Train: Loss=0.0042 Batch_id=937 Accuracy=99.08: 100%|██████████| 938/938 [00:45<00:00, 20.57it/s]
Test set: Average loss: 0.0129, Accuracy: 9957/10000 (99.57%)

Adjusting learning rate of group 0 to 1.0000e-04.
Epoch 19
Train: Loss=0.2548 Batch_id=937 Accuracy=99.04: 100%|██████████| 938/938 [00:45<00:00, 20.84it/s]
Test set: Average loss: 0.0132, Accuracy: 9954/10000 (99.54%)

Adjusting learning rate of group 0 to 1.0000e-04.
Epoch 20
Train: Loss=0.0566 Batch_id=937 Accuracy=99.08: 100%|██████████| 938/938 [00:43<00:00, 21.37it/s]
Test set: Average loss: 0.0131, Accuracy: 9958/10000 (99.58%)

Adjusting learning rate of group 0 to 1.0000e-04.
```  
## How to setup
### Prerequisits
```
1. python 3.8 or higher
2. pip 22 or higher
```

It's recommended to use virtualenv so that there's no conflict of package versions if there are multiple projects configured on a single system. 
Read more about [virtualenv](https://virtualenv.pypa.io/en/latest/). 

Once virtualenv is activated (or otherwise not opted), install required packages using following command. 

```
pip install requirements.txt
```

## Running IPython Notebook using jupyter
To run the notebook locally -
```
$> cd <to the project folder>
$> jupyter notebook
```
The jupyter server starts with the following output -
```
To access the notebook, open this file in a browser:
        file:///<path to home folder>/Library/Jupyter/runtime/nbserver-71178-open.html
    Or copy and paste one of these URLs:
        http://localhost:8888/?token=64bfa8105e212068866f24d651f51d2b1d4cc6e2627fad41
     or http://127.0.0.1:8888/?token=64bfa8105e212068866f24d651f51d2b1d4cc6e2627fad41
```

Open the above link in your favourite browser, a page similar to below shall be loaded.

![Jupyter server index page](https://github.com/piygr/s5erav1/assets/135162847/40087757-4c99-4b98-8abd-5c4ce95eda38)

- Click on the notebook (.ipynb) link.

A page similar to below shall be loaded. Make sure, it shows *trusted* in top bar. 
If it's not _trusted_, click on *Trust* button and add to the trusted files.

![Jupyter notebook page](https://github.com/piygr/s5erav1/assets/135162847/7858da8f-e07e-47cd-9aa9-19c8c569def1)
Now, the notebook can be operated from the action panel.

Happy Modeling :-) 
 
