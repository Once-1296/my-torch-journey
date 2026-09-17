import torch
import torch.nn as nn

class PixelArtGenerator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=5, embed_dim=50):
        super().__init__()
        
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        input_dim = noise_dim + embed_dim
        
        self.network = nn.Sequential(
            # 1x1 -> 4x4
            nn.ConvTranspose2d(input_dim, 256, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh() 
        )

    def forward(self, noise, labels):
        c = self.label_embedding(labels)
        
        noise = noise.view(noise.size(0), noise.size(1), 1, 1)
        c = c.view(c.size(0), c.size(1), 1, 1)
        
        x = torch.cat([noise, c], dim=1)
        return self.network(x)

if __name__ == "__main__":
    gen = PixelArtGenerator()
    dummy_noise = torch.randn(1, 100)
    dummy_label = torch.tensor([2])
    print("Generator Output Shape:", gen(dummy_noise, dummy_label).shape)