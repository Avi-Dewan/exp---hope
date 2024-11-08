import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch_fidelity import calculate_metrics
import numpy as np
from data.cifarloader import CIFAR10Loader

# Constants
z_dim = 128
img_channels = 3
batch_size = 128
lr = 0.0002
beta1 = 0.5
num_epochs = 5
save_interval = 2

# Directory setup
output_dir = './gan'
sample_dir = os.path.join(output_dir, 'samples')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(sample_dir, exist_ok=True)

# Define the Generator
class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)


# Define the Discriminator
# Updated Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False),  # Output: (batch_size, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # Output: (batch_size, 128, 8, 8)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # Output: (batch_size, 256, 4, 4)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # Output: (batch_size, 512, 2, 2)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            
            # Adjusted kernel size to (2, 2) here to handle (2, 2) input size
            nn.Conv2d(512, 1024, 2, 1, 0, bias=False),  # Output: (batch_size, 1024, 1, 1)
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(1024, 1, 1, 1, 0, bias=False),  # Output: (batch_size, 1, 1, 1)
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x).view(-1, 1)

# train_loader = CIFAR10Loader(root='./datasets', batch_size=128, split='train', aug=, shuffle=True, target_list=range(5, 10))
train_loader = CIFAR10Loader(root='./datasets', batch_size=batch_size, split='train', aug=None, shuffle=True, target_list=range(5, 10))


# Initialize models
G = Generator(z_dim=z_dim).cuda()
D = Discriminator().cuda()

# Print models and total parameter count
print("Generator Model:\n", G)
print("Total Parameters in Generator:", sum(p.numel() for p in G.parameters()))
print("\nDiscriminator Model:\n", D)
print("Total Parameters in Discriminator:", sum(p.numel() for p in D.parameters()))


# Loss and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

# Noise function
def get_noise(batch_size, z_dim):
    return torch.randn(batch_size, z_dim, 1, 1).cuda()

# FID calculation function
def compute_fid(real_images, fake_images):
    real_images = (real_images + 1) / 2  # Rescale to [0, 1]
    fake_images = (fake_images + 1) / 2  # Rescale to [0, 1]
    real_images = real_images.cpu().float()
    fake_images = fake_images.cpu().float()
    
    # Specify `fid=True` explicitly
    metrics = calculate_metrics(input1=real_images, input2=fake_images, fid=True)
    return metrics['frechet_inception_distance']

# Trackers
fid_scores = []
g_losses = []
d_losses = []

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (real_images, _, idx) in enumerate(tqdm(train_loader)): # Assuming CIFAR-10 dataloader is set up for classes 5-9
        real_images = real_images.cuda()
        batch_size = real_images.size(0)
        
        # Labels
        real_labels = torch.ones(batch_size, 1).cuda()
        fake_labels = torch.zeros(batch_size, 1).cuda()
        
        # Train Discriminator
        optimizer_D.zero_grad()
        
        # Real images
        real_preds = D(real_images).view(-1, 1)
        real_loss = criterion(real_preds, real_labels)
        
        # Fake images
        noise = get_noise(batch_size, z_dim)
        fake_images = G(noise)
        fake_preds = D(fake_images.detach()).view(-1, 1)
        fake_loss = criterion(fake_preds, fake_labels)
        
        # Backprop Discriminator
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        noise = get_noise(batch_size, z_dim)
        fake_images = G(noise)
        fake_preds = D(fake_images).view(-1, 1)
        g_loss = criterion(fake_preds, real_labels)
        g_loss.backward()
        optimizer_G.step()

        # Record losses
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
    
    # Print loss progress
    print(f"Epoch [{epoch}/{num_epochs}] | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}")

    # Generate and save images every few epochs
    if epoch % save_interval == 0:
        with torch.no_grad():
            fixed_noise = get_noise(20, z_dim)
            fake_images = G(fixed_noise).cpu()
            grid_img = vutils.make_grid(fake_images, nrow=5, normalize=True)
            vutils.save_image(grid_img, os.path.join(sample_dir, f"generated_images_epoch_{epoch}.png"))
        
# Save Generator and Discriminator
torch.save(G.state_dict(), os.path.join(output_dir, "generator.pth"))
torch.save(D.state_dict(), os.path.join(output_dir, "discriminator.pth"))
print("Models saved successfully.")

# Plot Losses
plt.figure(figsize=(6, 6))
plt.plot(g_losses, label="Generator Loss", color="blue")
plt.plot(d_losses, label="Discriminator Loss", color="red")
plt.title("Generator and Discriminator Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(output_dir, "loss_plot.png"))


# Plot FID over epochs
# plt.figure(figsize=(6, 6))
# plt.plot(range(len(fid_scores)), fid_scores, label="FID Score", color="green")
# plt.title("FID Score over Epochs")
# plt.xlabel("Epochs")
# plt.ylabel("FID Score")
# plt.legend()
# plt.savefig("fid_plot.png")
# plt.show()
