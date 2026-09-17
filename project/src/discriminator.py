import torch
import torch.nn as nn
import torch.nn.utils as utils

class PixelArtDiscriminator(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        
        # 1. Label Embedding 
        # We expand the label to exactly match the 64x64 image size
        self.label_embedding = nn.Embedding(num_classes, 64 * 64)
        
        # 2. The CNN (Shrinks 64x64 down to a single True/False probability)
        # Input channels = 3 (RGB image) + 1 (Label) = 4
        self.network = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(4, 32, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            utils.spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)),
            # nn.BatchNorm2d(64), <-- REMOVE THIS
            nn.LeakyReLU(0.2, inplace=True),
            
            utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            # nn.BatchNorm2d(128), <-- REMOVE THIS
            nn.LeakyReLU(0.2, inplace=True),
            
            utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            # nn.BatchNorm2d(256), <-- REMOVE THIS
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid() 
        )

    def forward(self, image, labels):
        # 1. Get label embedding and reshape to match image width/height (1 channel, 64x64)
        c = self.label_embedding(labels)
        c = c.view(c.size(0), 1, 64, 64)
        
        # 2. Stack the image and the label together (Creates the 4-channel input)
        x = torch.cat([image, c], dim=1) 
        
        # 3. Pass through the network
        output = self.network(x)
        
        # 4. Flatten the 1x1 output to a simple 1D probability
        return output.view(-1, 1)

if __name__ == "__main__":
    # Test the Discriminator
    dummy_image = torch.randn(1, 3, 64, 64) # 1 fake RGB image
    dummy_label = torch.tensor([2])         # Label 2 (e.g., a Rose)
    
    disc = PixelArtDiscriminator()
    prediction = disc(dummy_image, dummy_label)
    
    # Should print: torch.Size([1, 1]) and a number between 0 and 1
    print("Prediction Shape:", prediction.shape)
    print("Probability (Real=1, Fake=0):", prediction.item())