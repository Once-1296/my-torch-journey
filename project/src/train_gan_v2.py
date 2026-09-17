import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed_v3"
PROGRESS_DIR = ROOT / "data" / "gan_progress_v3"
CHECKPOINT_PATH = ROOT / "data" / "gan_checkpoint_3.pth"

from generator_v2 import PixelArtGenerator
from discriminator_v2 import PixelArtDiscriminator

# Differentiable Augmentation: Shifts the image around the canvas
def diff_translation(x, max_shift=8):
    batch_size = x.shape[0]
    # Generate random shifts for x and y
    shift_x = torch.randint(-max_shift, max_shift + 1, (batch_size,))
    shift_y = torch.randint(-max_shift, max_shift + 1, (batch_size,))
    
    x_aug = torch.zeros_like(x)
    for i in range(batch_size):
        # roll wraps pixels around. Since backgrounds are black, it just shifts the flower!
        x_aug[i] = torch.roll(x[i], shifts=(shift_y[i].item(), shift_x[i].item()), dims=(1, 2))
    return x_aug

def train():
    batch_size = 64
    epochs = 1500  # Increased for augmentation
    noise_dim = 100
    lr_gen = 0.0002
    lr_disc = 0.0001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
    ])
    
    dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    gen = PixelArtGenerator(noise_dim=noise_dim, num_classes=5).to(device)
    disc = PixelArtDiscriminator(num_classes=5).to(device)
    
    opt_gen = optim.Adam(gen.parameters(), lr=lr_gen, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=lr_disc, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # --- CHECKPOINT LOADING ---
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Found checkpoint at {CHECKPOINT_PATH}. Loading...")
        checkpoint = torch.load(CHECKPOINT_PATH)
        gen.load_state_dict(checkpoint['gen_state_dict'])
        disc.load_state_dict(checkpoint['disc_state_dict'])
        opt_gen.load_state_dict(checkpoint['opt_gen_state_dict'])
        opt_disc.load_state_dict(checkpoint['opt_disc_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}...")
    else:
        print("Starting fresh training...")

    for epoch in range(start_epoch, epochs):
        for batch_idx, (real_images, labels) in enumerate(dataloader):
            real_images = real_images.to(device)
            labels = labels.to(device)
            current_batch_size = real_images.size(0)
            
            # Labels (Smoothed for Disc, Pure 1.0 for Gen)
            real_targets = torch.full((current_batch_size, 1), 0.9).to(device)
            fake_targets = torch.zeros(current_batch_size, 1).to(device)
            gen_targets = torch.ones(current_batch_size, 1).to(device)
            
            # --- Train Discriminator ---
            opt_disc.zero_grad()
            
            # Apply Augmentation to Real
            real_images_aug = diff_translation(real_images)
            disc_real_pred = disc(real_images_aug, labels)
            loss_disc_real = criterion(disc_real_pred, real_targets)
            
            # Apply Augmentation to Fake
            noise = torch.randn(current_batch_size, noise_dim).to(device)
            fake_images = gen(noise, labels)
            fake_images_aug = diff_translation(fake_images)
            
            disc_fake_pred = disc(fake_images_aug.detach(), labels) 
            loss_disc_fake = criterion(disc_fake_pred, fake_targets)
            
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            loss_disc.backward()
            opt_disc.step()
            
            # --- Train Generator ---
            opt_gen.zero_grad()
            
            # We want Discriminator to think the Augmented Fakes are Real
            disc_fake_pred_for_gen = disc(fake_images_aug, labels)
            loss_gen = criterion(disc_fake_pred_for_gen, gen_targets)
            
            loss_gen.backward()
            opt_gen.step()
            
        print(f"Epoch [{epoch}/{epochs}] | Loss D: {loss_disc.item():.4f}, Loss G: {loss_gen.item():.4f}")
        
        # Save images and checkpoint every 5 epochs
        if epoch % 5 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                test_noise = torch.randn(25, noise_dim).to(device)
                test_labels = torch.tensor([i % 5 for i in range(25)]).to(device)
                test_images = gen(test_noise, test_labels)
                test_images = (test_images + 1) / 2 
                save_image(test_images, PROGRESS_DIR / f"epoch_{epoch}.png", nrow=5)
            
            # --- SAVE CHECKPOINT ---
            torch.save({
                'epoch': epoch,
                'gen_state_dict': gen.state_dict(),
                'disc_state_dict': disc.state_dict(),
                'opt_gen_state_dict': opt_gen.state_dict(),
                'opt_disc_state_dict': opt_disc.state_dict(),
            }, CHECKPOINT_PATH)

if __name__ == "__main__":
    train()