# implement the below functions:

# manual part
# --------------
# model prediction
# loss = MSE
# gradient with respect to weight
# training (lr = 0.01, n_iters = 10,update function and output training details)

# pytorch part
# --------------
# replace the gradient computation of the manual part with pytorch's autograd


#%%
# manual part
import numpy as np

# model prediction
def forward(w, x):
    return w * x

# loss = MSE
def loss(y_true, y_pred):
    return ((y_pred - y_true) ** 2).mean()

# gradient of loss function with respect to weight
def gradient(x, y_true, y_pred):
    return (2 * x * (y_pred - y_true)).mean()

X = np.array([1, 2, 3, 4], dtype = np.float32)
Y = np.array([2, 4, 6, 8], dtype = np.float32)

# initial w
w = 0.0

# training
learning_rate = 0.01
n_iters = 70

print(f"Prediction before training where x = {5} : {forward(w,5):.3f}")
for epoach in range(1,n_iters+1):

    # predict = forward pass
    y_pred = forward(w, X)
    loss_value = loss(Y, y_pred)

    print(f"epoach {epoach}: weight = {w:.3f}, loss = {loss_value:.8f}")

    # calculate gradient = backward
    gradient_value = gradient(X, Y, y_pred)

    # update weight
    w -= learning_rate * gradient_value

print(f"Prediction after training where x = {5}: {forward(w,5):.3f}")

# %%
# pytorch part
import torch

X_tor = torch.tensor([1, 2, 3, 4], dtype = torch.float32)
Y_tor = torch.tensor([2, 4, 6, 8], dtype = torch.float32)

# initial w
w_tor = torch.tensor(0, dtype = torch.float32,requires_grad= True)

# training
learning_rate = 0.01
n_iters = 70

# replace the gradient computation of the manual part with pytorch's autograd
print(f"Prediction before training where x = {5} : {forward(w_tor,5):.3f}")
for epoach in range(1,n_iters+1):

    # predict = forward pass
    y_pred_tor = forward(w_tor, X_tor)
    loss_value_tor = loss(Y_tor, y_pred_tor)

    print(f"epoach {epoach}: weight = {w_tor:.3f}, loss = {loss_value_tor:.8f}")

    # calculate gradient = backward
    loss_value_tor.backward()

    # update weight
    with torch.no_grad():
        w_tor -= learning_rate * w_tor.grad

    # zero the gradients after updating
    w_tor.grad.zero_()
    
print(f"Prediction after training where x = {5}: {forward(w_tor,5):.3f}")
