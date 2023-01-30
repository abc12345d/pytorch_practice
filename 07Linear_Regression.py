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

#%%
import torch
import torch.nn as nn
import numpy as np                  # for data transformation
from sklearn import datasets        # generate a regression dataset
import matplotlib.pyplot as plt     # to plt the data

#%%
# prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y.unsqueeze_(1)

# set up linear regression model
in_features = X.shape[1]
out_features = y.shape[1]
model = nn.Linear(in_features, out_features)

# set MSE as loss function 
loss = nn.MSELoss()

# set up SGD asoptimser 
learning_rate = 0.05
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training
n_iter = 100
for epoach in range(n_iter):
    # forward
    y_pred = model(X)
    loss_value = loss(y,y_pred)

    # backward
    loss_value.backward()

    # update
    optimiser.step()

    # zero the gradient before next iteration
    optimiser.zero_grad()

    print(f"epoach: {epoach}, loss: {loss_value:.4f}")


# %%
# plot y_true and y_pred on same scale
predicted = y_pred.detach().numpy()

plt.plot(X, y, 'bo')
plt.plot(X, predicted, 'r')
plt.show()
