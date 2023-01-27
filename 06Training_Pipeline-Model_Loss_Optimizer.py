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