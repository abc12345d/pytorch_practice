import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets # to load a binary classification dataset
from sklearn.preprocessing import StandardScalar # to scale our feature
from sklearn.model_selection import train_test_split # to separate training and testing data

# TODO
# (0) prepare data
#     - load_breast_cancer dataset
#     - split train and test data
#     - scale features so that the features have zero mean and unit variance
#     - transform to tensor where type = float32
# (1) set up model
#     - f = wx + b, sigmoid at the end 
#     - create custom logistic regression model
# (2) loss and optimiser
#     - use nn.BCE (binary cross entropy loss)
#     - SGD (stochastic gradient descent) optimiser
# (3) training loop
#     - forward pass and loss
#     - backward pass
#     - update parameters
#     - output information about training
# (4) evaluate the model 
#     - calculate accuracy
