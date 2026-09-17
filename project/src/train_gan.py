import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed_v2"
PROGRESS_DIR = ROOT / "data" / "gan_progress"

# Import your architectures
from generator import PixelArtGenerator
from discriminator import PixelArtDiscriminator

def train():
    # 1. Hyperparameters & Setup
    batch_size = 64
    epochs = 400  # Let it cook! Since it's fast, give it time to sculpt the shapes.
    noise_dim = 100
    lr_gen = 0.0002    # Generator stays the same
    lr_disc = 0.0001   # Discriminator learns slower
    
    # Check for GPU (Training GANs on CPU works but is slow)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Create folder to save training progress images
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    
    # 2. Dataset Loading
    # Tanh in Generator outputs [-1, 1], so we must normalize our dataset to [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
    ])
    
    # Point this to the root folder holding your subfolders (roses, tulips, etc.)
    dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Initialize Networks & Optimizers
    gen = PixelArtGenerator(noise_dim=noise_dim, num_classes=5).to(device)
    disc = PixelArtDiscriminator(num_classes=5).to(device)
    
    # Standard GAN optimizers use Adam with a lower momentum (beta1=0.5)
    opt_gen = optim.Adam(gen.parameters(), lr=lr_gen, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=lr_disc, betas=(0.5, 0.999))
    criterion = nn.BCELoss() # Binary Cross Entropy (for 1 vs 0 prediction)
    
    # 4. The Training Loop
    print("Starting Training Loop...")
    for epoch in range(epochs):
        for batch_idx, (real_images, labels) in enumerate(dataloader):
            real_images = real_images.to(device)
            labels = labels.to(device)
            current_batch_size = real_images.size(0)
            
            # Create labels for loss calculation (Real = 1s, Fake = 0s)
            # Old: real_targets = torch.ones(current_batch_size, 1).to(device)
            
            # New: Label Smoothing (Set real targets to 0.9 instead of 1.0)
            real_targets = torch.full((current_batch_size, 1), 0.9).to(device)
            fake_targets = torch.zeros(current_batch_size, 1).to(device)
            
            # ==========================================
            # STEP 1: Train Discriminator
            # ==========================================
            opt_disc.zero_grad()
            
            # A. Real Images
            disc_real_pred = disc(real_images, labels)
            loss_disc_real = criterion(disc_real_pred, real_targets)
            
            # B. Fake Images
            noise = torch.randn(current_batch_size, noise_dim).to(device)
            fake_images = gen(noise, labels)
            disc_fake_pred = disc(fake_images.detach(), labels) # Detach so we don't calculate Generator gradients yet
            loss_disc_fake = criterion(disc_fake_pred, fake_targets)
            
            # Combine and Update
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            loss_disc.backward()
            opt_disc.step()
            
            # ==========================================
            # STEP 2: Train Generator
            # ==========================================
            opt_gen.zero_grad()
            
            # Create a target of 1.0 specifically for the Generator
            gen_targets = torch.ones(current_batch_size, 1).to(device)
            
            # Predict fakes again...
            disc_fake_pred_for_gen = disc(fake_images, labels)
            loss_gen = criterion(disc_fake_pred_for_gen, gen_targets) # Use 1.0 targets here
            
            loss_gen.backward()
            opt_gen.step()
            
        # 5. Print progress and save sample images every 5 epochs
        print(f"Epoch [{epoch+1}/{epochs}] | Loss D: {loss_disc.item():.4f}, Loss G: {loss_gen.item():.4f}")
        
        if epoch % 5 == 0:
            with torch.no_grad():
                # Test generating a few of each class (0 to 4)
                # Instead of standard normal noise:
                # test_noise = torch.randn(25, noise_dim).to(device)

                # Use truncated noise (e.g., between -1 and 1)
                test_noise = torch.randn(25, noise_dim).clamp(-1, 1).to(device)
                test_labels = torch.tensor([i % 5 for i in range(25)]).to(device)
                test_images = gen(test_noise, test_labels)
                
                # Un-normalize back to [0, 1] for saving
                test_images = (test_images + 1) / 2 
                save_image(test_images, PROGRESS_DIR / f"epoch_{epoch}.png", nrow=5)
                
    # 6. Save the final model (We only need the Generator!)
    torch.save(gen.state_dict(), "pixel_generator.pth")
    print("Training Complete! Generator saved.")

if __name__ == "__main__":
    train()