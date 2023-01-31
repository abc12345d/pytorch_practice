import torch
import torchvision
import numpy as np
# TODO: 
# (1) implement ToTensor class which convert ndarrays to Tensors
#     - have __call__(self,sample)
# (2) implement MulTransform class which multiply inputs with a given factor 
#     - have __init__(self,factor) and __call__(self,sample)
# (3) output data (hint: transform param of Dataset and torchvision.transforms.Compose())
#     - without transform
#     - with toTensor transform 
#     - with toTensor and Multiplication transform