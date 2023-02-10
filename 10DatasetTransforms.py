#%%
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset

path_wine_data = "./data/wine.csv"
# TODO: 
# (1) implement ToTensor class which convert ndarrays to Tensors
#     - have __call__(self,sample)
# (2) implement MulTransform class which multiply inputs with a given factor 
#     - have __init__(self,factor) and __call__(self,sample)
# (3) output data (hint: transform param of Dataset and torchvision.transforms.Compose())
#     - without transform
#     - with toTensor transform 
#     - with toTensor and Multiplication transform
class ToTensor():
    # Convert ndarrays to Tensors
    def __call__(self, sample):
        y, x = sample
        return torch.from_numpy(y), torch.from_numpy(x)

class MulTransform():
    # multiply inputs with a given factor
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self,sample):
        y, x = sample
        return y, x * self.factor

class Wine_Dataset(Dataset):
    def __init__(self, transform = None):
        super().__init__()
        wine_data = np.loadtxt(path_wine_data, delimiter=",",dtype= np.float32, skiprows=1)
        
        self.y = wine_data[:, [0]]
        self.x = wine_data[:, 1:]
        self.no_samples = self.y.shape[0]
        self.transform = transform


    def __getitem__(self, index):
        sample = self.y[index], self.x[index]
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.no_samples


npToTensor = ToTensor()
mulTrans = MulTransform(2)
composed = torchvision.transforms.Compose([npToTensor,mulTrans])

wine_data_without = Wine_Dataset()
print(f"without: {wine_data_without[0]}")
wine_data_npToTensor = Wine_Dataset(transform = npToTensor)
print(f"toTensor: {wine_data_npToTensor[0]}")
wine_data_mulTrans = Wine_Dataset(transform = mulTrans)
print(f"mulTrans: {wine_data_mulTrans[0]}")
wine_data_composed = Wine_Dataset(transform = composed)
print(f"composed: {wine_data_composed[0]}")
