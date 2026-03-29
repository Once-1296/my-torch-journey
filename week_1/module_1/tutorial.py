import torch

# 1. Creating Tensors
x = torch.tensor([2.0, 3.0])
w = torch.tensor([0.5, 0.2], requires_grad=True) # requires_grad=True tells PyTorch to track operations on this tensor
b = torch.tensor(0.1, requires_grad=True)

# 2. A simple operation (Forward Pass)
# Dot product of x and w, plus bias
y_pred = torch.dot(w, x) + b 
print(f"Prediction: {y_pred.item()}")

# 3. Assume the true value we wanted was 2.0
y_true = torch.tensor(2.0)
loss = (y_pred - y_true)**2 # Mean Squared Error
print(f"Loss: {loss.item()}")

# 4. The Magic: Autograd (Backward Pass)
loss.backward() # Computes the gradient of the loss with respect to all tensors with requires_grad=True

# Print the gradients
print(f"Gradient of w: {w.grad}")
print(f"Gradient of b: {b.grad}")