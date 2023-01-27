import torch
import torch.nn as nn
import numpy as np                  # for data transformation
from sklearn import datasets        # generate a regression dataset
import matplotlib.pyplot as plt     # to plt the data

# TODO
# (0) prepare data 
#     - using datasets.make_regression 
#     - transform the generated data to tensor
# (1) set up model 
# (2) loss and optimiser
# (3) training loop
#     - forward & loss
#     - backward
#     - update
#     - output information about training
# (4) plot the y_true and y_pred on same scale