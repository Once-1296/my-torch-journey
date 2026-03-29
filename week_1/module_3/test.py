import torch
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

# Let's peek at the first batch
images, labels = next(iter(train_loader))
print(f"Image batch shape: {images.shape}") # Output: [32, 1, 28, 28] 
print(f"Labels batch shape: {labels.shape}") # Output: [32]

class FashionClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_layer = torch.nn.Linear(in_features=784, out_features=10)

    def forward(self,x):
        return self.linear_layer(self.flatten(x))
    
model = FashionClassifier()
predicted_labels = model.forward(images)
print(predicted_labels.shape)