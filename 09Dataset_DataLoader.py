import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

path_wine_data = "./data/wine.csv"

# gradient computation etc. not efficient for whole data set
# -> divide dataset into small batches
# TODO: 
# (1) create custom Dataset object for wine data which inherit Dataset and implement the below methods
#     - __init__(self)            # data loading
#     - __getitem__(self,index)   # so that we can dataset[index]
#     - __len__(self)             # so that we can len(dataset)
# (2) initialise a DataLoader where dataset = the instance of our custom Dataset object
#     - try iterate the DataLoader
