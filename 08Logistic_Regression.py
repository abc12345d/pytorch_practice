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

#%%
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets # to load a binary classification dataset
from sklearn.preprocessing import StandardScaler # to scale our feature
from sklearn.model_selection import train_test_split # to separate training and testing data

# create custom logistic regression model
class LogisticRegresssion(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.Linear = nn.Linear(in_dim,out_dim)

    def forward(self, X):
        y_pred = torch.sigmoid(self.Linear(X))
        return y_pred

#%%
# prepare data
breast_cancer_dataset = datasets.load_breast_cancer()
X, y = breast_cancer_dataset.data, breast_cancer_dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 1234)

scaler = StandardScaler()
# perform data normalisation on training data
X_train = scaler.fit_transform(X_train)
# perform data normalisation on testing data using the mean and variance of training data
X_test = scaler.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1)
y_test = torch.from_numpy(y_test.astype(np.float32)).unsqueeze(1)

# set up model
in_features = X_train.shape[1]
out_features = y_train.shape[1]
model = LogisticRegresssion(in_features,out_features)

# set nn.BCE (binary cross entropy loss) as loss function
loss = nn.BCELoss()

# set SGD (stochastic gradient descent) as optimiser
learning_rate = 0.01
optimiser = torch.optim.SGD(model.parameters(),lr = learning_rate)

# training
no_epochs = 100
for epoch in range(no_epochs):
    # forward
    y_pred = model(X_train)
    loss_value = loss(y_pred,y_train)

    # backward
    loss_value.backward()

    # update 
    optimiser.step()

    # zero gradients before next iteration
    optimiser.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f"epoch: {epoch+1}, loss: {loss_value:.4f}")

# %%
with torch.no_grad():
    y_test_pred = model(X_test)
    # convert value from interval [0,1] to either 0 or 1
    y_test_pred = y_test_pred.round()

    # calculate accuracy rate
    acc = y_test_pred.eq(y_test).sum()/y_test.shape[0]
    print(f'accuracy: {acc.item():.4f}')
