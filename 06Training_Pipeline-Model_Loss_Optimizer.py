# General training pipeline in pytorch
# (1) Design model (no of input, no of output, forward pass)
# (2) Construct the loss and optimser
# (3) Training loop
#     - forward pass: compute prediction
#     - backward pass: gradients
#     - update weights

# TODO:  replace the manual part by corresponding (pytorch) part
# Step 4: prediction (PyTorch Model - Linear)
# Step 2: Gradients computation (Autograd)
# Step 3: Loss Computation (PyTorch Loss - MSE)
# Step 3: Paramater updates (PyTorch Optimizer - SGD)

# TODO: write a custom linear regression model 
# must be derived from nn.Module

#%%
import torch 
import torch.nn as nn

# custom linear regression model
class LinearRegression(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim,out_dim)
    
    def forward(self, x):
        return self.linear(x)

#%%
#######################################
##--------------step3----------------##
#######################################

# model prediction
def forward(w, x):
    return w * x

X = torch.tensor([1, 2, 3, 4], dtype = torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype = torch.float32)

# initial w
w = torch.tensor(0, dtype = torch.float32, requires_grad = True)

# define loss function as PyTorch MSE
loss = nn.MSELoss()

# training
learning_rate = 0.01
n_iters = 70

# define PyTorch SGD for parameter update
optimiser = torch.optim.SGD([w],lr = learning_rate)

print(f"Prediction before training where x = {5} : {forward(w,5):.3f}")
for epoach in range(1,n_iters+1):

    # predict = forward pass
    y_pred = forward(w, X)
    loss_value = loss(Y, y_pred)

    print(f"epoach {epoach}: weight = {w:.3f}, loss = {loss_value:.8f}")

    # calculate gradient = backward
    loss_value.backward()

    optimiser.step()

    # zero the gradient
    w.grad.zero_()


#%%
#######################################
##--------------step4----------------##
#######################################

X = torch.tensor([1, 2, 3, 4], dtype = torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype = torch.float32)
X_test = torch.tensor([[5]], dtype = torch.float32)

# transform data for nn.Linear() so that data.size() = (no_samples,no_features)
X.unsqueeze_(1)
Y.unsqueeze_(1)
in_feature = X.size()[1]
out_feature = Y.size()[1]

# PyTorch Linear model
# model = nn.Linear(in_features = in_feature, out_features = out_feature)

# custom Linear model
model = LinearRegression(in_dim = in_feature, out_dim = out_feature)

# define loss function as PyTorch MSE
loss = nn.MSELoss()

# training
learning_rate = 0.05
n_iters = 100

# define PyTorch SGD for parameter update
optimiser = torch.optim.SGD(model.parameters(),lr = learning_rate)

print(f"Prediction before training where x = {5} : {model(X_test).item():.3f}")

for epoach in range(1,n_iters+1):

    # predict = forward pass
    y_pred = model(X)
    loss_value = loss(Y, y_pred)

    print(f"epoach {epoach}: weight = {w:.3f}, loss = {loss_value.item():.8f}")

    # calculate gradient = backward
    loss_value.backward()

    optimiser.step()

    # zero the gradient
    optimiser.zero_grad()

print(f"Prediction after training where x = {5}: {model(X_test).item():.3f}")

# %%
