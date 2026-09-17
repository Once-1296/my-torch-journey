import torch
import torch.nn as nn

class PixelArtGenerator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=5, embed_dim=50):
        super().__init__()
        
        # 1. The Label Embedding
        # Turns class (0-4) into a vector of size 50
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        
        # Total input size = 100 (noise) + 50 (label) = 150
        input_dim = noise_dim + embed_dim
        
        # 2. The Transposed CNN (Expands to 64x64)
        self.network = nn.Sequential(
            # Input: 150 channels, 1x1 size. 
            # Output: 256 channels, 4x4 size.
            nn.ConvTranspose2d(input_dim, 256, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # Input: 256 channels, 4x4 size.
            # Output: 128 channels, 8x8 size.
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # Input: 128 channels, 8x8 size.
            # Output: 64 channels, 16x16 size.
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Input: 64 channels, 16x16 size.
            # Output: 32 channels, 32x32 size.
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # Input: 32 channels, 32x32 size.
            # Output: 3 channels (RGB), 64x64 size.
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            # Tanh scales the output pixel colors to be between -1 and 1
            nn.Tanh() 
        )

    def forward(self, noise, labels):
        # 1. Get the embedding for the label
        c = self.label_embedding(labels) # Shape: [batch_size, embed_dim]
        
        # 2. Reshape noise and condition so they have 1x1 spatial dimensions
        noise = noise.view(noise.size(0), noise.size(1), 1, 1)
        c = c.view(c.size(0), c.size(1), 1, 1)
        
        # 3. Concatenate them together along the channel dimension
        x = torch.cat([noise, c], dim=1) # Shape: [batch_size, 150, 1, 1]
        
        # 4. Generate the image!
        return self.network(x)

if __name__ == "__main__":
    # Create fake inputs
    dummy_noise = torch.randn(1, 100) # 1 image, 100-dimensional random noise
    dummy_label = torch.tensor([2])   # Label 2 (e.g., a Rose)
    
    # Initialize Generator
    gen = PixelArtGenerator()
    
    # Generate fake image
    fake_image = gen(dummy_noise, dummy_label)
    
    # Should print: torch.Size([1, 3, 64, 64])
    print("Generated Image Shape:", fake_image.shape)

