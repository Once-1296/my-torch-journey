import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 1. A Deeper Model using nn.Sequential
class DeepFashionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        
        # nn.Sequential is a clean way to stack layers in order
        self.network = nn.Sequential(
            nn.Linear(784, 128),  # Hidden Layer 1: 784 inputs -> 128 neurons
            nn.ReLU(),            # Activation
            nn.Linear(128, 64),   # Hidden Layer 2: 128 inputs -> 64 neurons
            nn.ReLU(),            # Activation
            nn.Linear(64, 10)     # Output Layer: 64 inputs -> 10 categories
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

# 2. The Right Loss Function
# For multi-class classification, ALWAYS use CrossEntropyLoss.
# It automatically applies Softmax (turns raw scores into probabilities) 
# and calculates the error.
loss_fn = nn.CrossEntropyLoss()

# dataloading
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Transforms: We must convert images to PyTorch Tensors
# transforms.ToTensor() converts pixel values from 0-255 into floats from 0.0-1.0
transform = transforms.ToTensor()

# 2. Download and load the training data
train_data = datasets.FashionMNIST(
    root="data",          # Where to save the files
    train=True,           # Get the training set
    download=True,        # Download if it's not already there
    transform=transform   # Apply our tensor conversion
)

# 3. Put it in a DataLoader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# 4. Model init

model = DeepFashionClassifier()

# 5. setting optimiser

from torch.optim import Adam as Adam
optimizer = Adam(model.parameters(), lr=0.005)

def train(model):
    model.train() # Set to training mode
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for x_batch, y_batch in train_loader:
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        
        # Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        predictions = y_pred.argmax(dim=1)
        total_correct += (predictions == y_batch).sum().item()
        total_samples += y_batch.size(0) # Count actual images
        
    return total_loss / len(train_loader), (total_correct / total_samples) * 100,model

def train_call(n=25, model=None):
    for i in range(n):
        loss,acc,model = train(model)
        if i % 5 == 4: print(f"Loss: {loss}, Accuracy: {acc} %")
    return model