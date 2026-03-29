import torch

# 1. Data Generation (Pure PyTorch)
# Create X of shape (100, 1) with values between 0 and 10
X = torch.rand(100, 1) * 10 
# True relationship: Y = 3X + 2 + noise
Y = 3 * X + 2 + torch.randn(100, 1) 

learning_rate = 0.005 # Lowered slightly so the model doesn't overcorrect

# 2. Weight Initialization
# We just need ONE weight and ONE bias. 
W = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

print(f"Initial: w: {W.item():.4f}, b: {b.item():.4f}\n")

# 3. Training Loop
for i in range(10000):
    # Forward pass: Element-wise multiplication
    y_pred = X * W + b
    
    # Calculate Mean Squared Error
    mse = torch.mean((Y - y_pred)**2) 
    
    # Backward pass: Compute gradients
    mse.backward()
    
    # Update weights without tracking gradients
    with torch.no_grad():
        W -= learning_rate * W.grad  # In-place update
        b -= learning_rate * b.grad  # In-place update
        
        # VERY IMPORTANT: Zero the gradients for the next loop!
        W.grad.zero_()
        b.grad.zero_()
    
    # Print progress
    if i % 200 == 199: 
        print(f"Epoch {i+1} | MSE Loss: {mse.item():.4f} | w: {W.item():.4f}, b: {b.item():.4f}")

print(f"\nFinal Expected -> w: ~3.0, b: ~2.0")
print(f"Final Actual   -> w: {W.item():.4f}, b: {b.item():.4f}")