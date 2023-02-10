#%%
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

class Wine_Dataset(Dataset):
    def __init__(self):
        super().__init__()
        wine_data = np.loadtxt(path_wine_data, delimiter=",",dtype= np.float32, skiprows=1)
        
        # first column is the class label, the rest are the features
        self.y = torch.from_numpy(wine_data[:, [0]]) # shape = (no_samples, n_features)
        self.X = torch.from_numpy(wine_data[:, 1:]) # shape = (no_samples, 1)
        self.no_samples = self.y.shape[0]

    def __getitem__(self, index):
        return self.y[index], self.X[index]

    def __len__(self):
        return self.no_samples

wine_data = Wine_Dataset()
batch_size = 20
wine_dataloader = DataLoader(dataset = wine_data, batch_size = batch_size, 
                            shuffle = True, num_workers = 2)

wine_iterator = iter(wine_dataloader)
data = next(wine_iterator)
features, labels = data
print(features, labels)

no_iter = 0
for batch in wine_dataloader:
    y, x = batch
    no_iter += 1

# %%
# test case: 
if (no_iter * batch_size) >= len(wine_dataloader):
    print(f"test case: {True}")
else:
    print(f"test case: {False}")



# %%
