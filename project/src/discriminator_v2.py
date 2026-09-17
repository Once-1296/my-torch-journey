import torch
import torch.nn as nn
import torch.nn.utils as utils

class PixelArtDiscriminator(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        
        self.label_embedding = nn.Embedding(num_classes, 64 * 64)
        
        self.network = nn.Sequential(
            # 64x64 -> 32x32
            utils.spectral_norm(nn.Conv2d(4, 32, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16 (No BatchNorm!)
            utils.spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8 (No BatchNorm!)
            utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4 (No BatchNorm!)
            utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final output layer
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid() 
        )

    def forward(self, image, labels):
        c = self.label_embedding(labels)
        c = c.view(c.size(0), 1, 64, 64)
        
        x = torch.cat([image, c], dim=1)
        output = self.network(x)
        return output.view(-1, 1)

if __name__ == "__main__":
    disc = PixelArtDiscriminator()
    dummy_image = torch.randn(1, 3, 64, 64)
    dummy_label = torch.tensor([2])
    print("Discriminator Output Shape:", disc(dummy_image, dummy_label).shape)