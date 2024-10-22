import torch
import torch.nn as nn
import torch.nn.functional as F

from .BigGAN import Generator as BigG
from .BigGAN import Discriminator as BigD

class Generator(nn.Module):
    def __init__(self, classes=10, latent_dim=100, img_size=32, channels=3):
        super().__init__()
        self.G = BigG(n_classes=classes, dim_z=latent_dim, resolution=img_size)

    def forward(self, noise, labels):
        # Concatenate label embedding and noise to form input
        img = self.G(noise, labels)
        return img

class Discriminator(nn.Module):
    def __init__(self, classes=10, img_size=32, channels=3):
        super().__init__()
        self.D = BigD(n_classes=classes, resolution=img_size)

    def forward(self, img, labels):
        # Concatenate image and label condition
        validity = self.D(img, labels)
        # validity = F.sigmoid(validity)
        return validity


