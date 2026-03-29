import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 1. The Data Pipeline
X = torch.rand(100, 1) * 10
Y = 3 * X + 2 + torch.randn(100, 1)

# TensorDataset pairs your X and Y together. 
# DataLoader shuffles it and serves it in batches of 10.
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 2. The Model Class
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Linear automatically creates the W and b tensors for us!
        self.linear_layer = nn.Linear(in_features=1, out_features=1)
        
    def forward(self, x):
        # This replaces our old "y_pred = X * W + b"
        return self.linear_layer(x)

# 3. Initialization
model = LinearRegressionModel()

# Built-in Mean Squared Error
loss_fn = nn.MSELoss() 

# Built-in Optimizer (Stochastic Gradient Descent)
# We pass it model.parameters() so it knows which weights to update
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def train():
    total_loss = 0
    for x_batch, y_batch in dataloader:
        # 1. Forward (call the model directly)
        y_pred = model(x_batch)
        
        # 2. Loss (predictions first)
        loss = loss_fn(y_pred, y_batch)
        
        # 3. Track loss safely using .item()
        total_loss += loss.item()
        
        # 4. Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Average the loss over the number of batches
    return total_loss / len(dataloader)
for i in range(50):
    loss = train()
    if i % 10 == 0: print(f"Loss; {loss}")

print(f"Final W: {model.linear_layer.weight.item()}, b: {model.linear_layer.bias.item()}")